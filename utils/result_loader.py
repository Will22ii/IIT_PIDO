# utils/result_loader.py

import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StageResult:
    """
    Container for stage result
    """
    stage: str
    problem_name: str
    csv_path: str
    pkl_path: Optional[str]
    df: pd.DataFrame
    metadata: Optional[dict]
    metadata_path: Optional[str]


class ResultLoader:
    """
    Load stage-based results produced by ResultSaver

    - CSV is mandatory
    - metadata is optional
    - latest result is selected by default
    """

    def __init__(self):
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    def _get_stage_dir(self, stage: str) -> str:
        return os.path.join(self.project_root, "result", stage.lower())

    def _find_latest_csv(self, stage: str, problem_name: str) -> str:
        stage_dir = self._get_stage_dir(stage)

        if not os.path.exists(stage_dir):
            raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

        prefix = f"{stage.lower()}_result_{problem_name}"
        candidates = [
            f for f in os.listdir(stage_dir)
            if f.startswith(prefix) and f.endswith(".csv")
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No CSV found for stage='{stage}', problem='{problem_name}'"
            )

        candidates.sort(reverse=True)
        return os.path.join(stage_dir, candidates[0])

    def _load_metadata(self, meta_path: str) -> Optional[dict]:
        if not meta_path or not os.path.exists(meta_path):
            return None
        with open(meta_path, "r") as f:
            return json.load(f)

    def _find_latest_metadata_v1(
        self,
        stage: str,
        problem_name: Optional[str] = None,
    ) -> Optional[str]:
        stage_dir = self._get_stage_dir(stage)
        if not os.path.exists(stage_dir):
            return None

        meta_files = [
            f for f in os.listdir(stage_dir)
            if f.startswith(f"{stage.lower()}_metadata_") and f.endswith(".json")
        ]
        if not meta_files:
            return None

        best_path = None
        best_time = None

        for name in meta_files:
            path = os.path.join(stage_dir, name)
            metadata = self._load_metadata(path)
            if not metadata:
                continue
            if problem_name and metadata.get("problem") != problem_name:
                continue
            created_at = metadata.get("created_at")
            if not created_at:
                continue
            try:
                created_dt = datetime.fromisoformat(created_at)
            except ValueError:
                continue
            if best_time is None or created_dt > best_time:
                best_time = created_dt
                best_path = path

        return best_path

    def _find_latest_metadata_v2(
        self,
        stage: str,
        problem_name: Optional[str] = None,
    ) -> Optional[str]:
        result_root = os.path.join(self.project_root, "result")
        if not os.path.exists(result_root):
            return None

        run_dirs = [
            d for d in os.listdir(result_root)
            if d.startswith("run_") and os.path.isdir(os.path.join(result_root, d))
        ]
        if not run_dirs:
            return None

        best_path = None
        best_time = None
        best_mtime = None

        for run_dir in run_dirs:
            meta_path = os.path.join(result_root, run_dir, stage, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            metadata = self._load_metadata(meta_path)
            if not metadata:
                continue
            if problem_name and metadata.get("problem") != problem_name:
                continue
            created_at = metadata.get("created_at")
            created_dt = None
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at)
                except ValueError:
                    created_dt = None
            if created_dt:
                if best_time is None or created_dt > best_time:
                    best_time = created_dt
                    best_path = meta_path
                    best_mtime = None
                continue
            mtime = os.path.getmtime(meta_path)
            if best_time is None and (best_mtime is None or mtime > best_mtime):
                best_mtime = mtime
                best_path = meta_path

        return best_path

    def _find_latest_metadata(
        self,
        stage: str,
        problem_name: Optional[str] = None,
    ) -> Optional[str]:
        v2_path = self._find_latest_metadata_v2(stage, problem_name)
        v1_path = self._find_latest_metadata_v1(stage, problem_name)

        if v2_path:
            return v2_path
        if v1_path:
            return v1_path
        return None

    def _resolve_metadata_path(
        self,
        stage: str,
        problem_name: Optional[str],
        metadata_path: Optional[str],
    ) -> Optional[str]:
        if metadata_path:
            if os.path.exists(metadata_path):
                return metadata_path
            print(f"[ResultLoader] Metadata path not found: {metadata_path}")
        meta_path = self._find_latest_metadata(stage, problem_name)
        if meta_path:
            source = "v2" if os.path.basename(meta_path) == "metadata.json" else "v1"
            print(
                f"[ResultLoader] No metadata_path provided; using latest ({source}): {meta_path}"
            )
        return meta_path

    def _extract_timestamp(self, filename: str) -> Optional[datetime]:
        stem = os.path.splitext(os.path.basename(filename))[0]
        parts = stem.split("_")
        if len(parts) < 3:
            return None
        date_part = parts[-2]
        time_part = parts[-1]
        try:
            return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    def _find_csv_for_metadata(self, stage: str, metadata: dict, meta_path: str) -> str:
        stage_dir = self._get_stage_dir(stage)
        csv_path = metadata.get("csv_path")
        if csv_path:
            if os.path.exists(csv_path):
                return csv_path
            print(f"[ResultLoader] CSV path not found: {csv_path}")

        artifacts = metadata.get("artifacts", {})
        results_csv = artifacts.get("results_csv")
        if results_csv:
            base_dir = os.path.dirname(meta_path)
            candidate = os.path.join(base_dir, results_csv)
            if os.path.exists(candidate):
                return candidate
            print(f"[ResultLoader] CSV path not found: {candidate}")

        if not os.path.exists(stage_dir):
            raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

        problem_name = metadata.get("problem")
        created_at = metadata.get("created_at")
        if not problem_name or not created_at:
            print(
                "[ResultLoader] Metadata missing 'problem' or 'created_at'; "
                "falling back to latest CSV."
            )
            return self._find_latest_csv(stage, problem_name or "")

        try:
            created_dt = datetime.fromisoformat(created_at)
        except ValueError:
            print(
                f"[ResultLoader] Invalid created_at in metadata: {created_at}; "
                "falling back to latest CSV."
            )
            return self._find_latest_csv(stage, problem_name)

        prefix = f"{stage.lower()}_result_{problem_name}"
        candidates = [
            os.path.join(stage_dir, f)
            for f in os.listdir(stage_dir)
            if f.startswith(prefix) and f.endswith(".csv")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No CSV found for stage='{stage}', problem='{problem_name}'"
            )

        best_path = None
        best_delta = None

        for path in candidates:
            ts = self._extract_timestamp(path)
            if not ts:
                continue
            delta = abs((created_dt - ts).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_path = path

        if best_path:
            return best_path

        print("[ResultLoader] No timestamp match; falling back to latest CSV.")
        candidates.sort(reverse=True)
        return candidates[0]

    def _find_pkl_for_metadata(self, stage: str, metadata: dict, meta_path: str) -> Optional[str]:
        if stage.upper() != "MODELER":
            return None

        stage_dir = self._get_stage_dir(stage)
        if not os.path.exists(stage_dir):
            raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

        model_path = metadata.get("model_path")
        if model_path:
            if os.path.exists(model_path):
                return model_path
            print(f"[ResultLoader] Model path not found: {model_path}")

        artifacts = metadata.get("artifacts", {})
        model_artifact = artifacts.get("model_path")
        if model_artifact:
            base_dir = os.path.dirname(meta_path)
            candidate = os.path.join(base_dir, model_artifact)
            if os.path.exists(candidate):
                return candidate
            print(f"[ResultLoader] Model path not found: {candidate}")

        problem_name = metadata.get("problem")
        created_at = metadata.get("created_at")
        if not problem_name or not created_at:
            return None

        try:
            created_dt = datetime.fromisoformat(created_at)
        except ValueError:
            return None

        candidates = [
            os.path.join(stage_dir, f)
            for f in os.listdir(stage_dir)
            if f.endswith(".pkl") and problem_name in f
        ]
        if not candidates:
            return None

        best_path = None
        best_delta = None

        for path in candidates:
            ts = self._extract_timestamp(path)
            if not ts:
                continue
            delta = abs((created_dt - ts).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_path = path

        if best_path:
            return best_path

        print("[ResultLoader] No PKL timestamp match; falling back to latest PKL.")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def load_stage(
        self,
        *,
        stage: str,
        problem_name: Optional[str] = None,
        csv_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        allow_latest_fallback: bool = True,
    ) -> StageResult:
        metadata = None

        if not allow_latest_fallback and not metadata_path:
            raise FileNotFoundError(
                "metadata_path is required when latest fallback is disabled."
            )

        meta_path = self._resolve_metadata_path(stage, problem_name, metadata_path)
        if not meta_path:
            raise FileNotFoundError(
                f"No metadata found for stage='{stage}'"
                + (f", problem='{problem_name}'" if problem_name else "")
            )
        metadata = self._load_metadata(meta_path)
        if not metadata:
            raise RuntimeError(f"Failed to load metadata: {meta_path}")

        if csv_path is None:
            csv_path = self._find_csv_for_metadata(stage, metadata, meta_path)
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")

        pkl_path = self._find_pkl_for_metadata(stage, metadata, meta_path)
        df = pd.read_csv(csv_path)

        return StageResult(
            stage=stage.upper(),
            problem_name=metadata.get("problem", problem_name or ""),
            csv_path=csv_path,
            pkl_path=pkl_path,
            df=df,
            metadata=metadata,
            metadata_path=meta_path,
        )
