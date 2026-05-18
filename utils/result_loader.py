# utils/result_loader.py

import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

TASK_DIR_MAP = {
    "CAE": "CAE",
    "DOE": "DOE",
    "MODELER": "Modeler",
    "EXPLORER": "Explorer",
    "OPT": "OPT",
}


@dataclass
class TaskResult:
    """Container for task result."""

    task: str
    problem_name: str
    csv_path: str
    pkl_path: Optional[str]
    df: pd.DataFrame
    metadata: Optional[dict]
    metadata_path: Optional[str]


class ResultLoader:
    """Load v3 task results from run directories only."""

    def __init__(self):
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    def _task_dir_name(self, task: str) -> str:
        task_key = str(task).strip().upper()
        if task_key in TASK_DIR_MAP:
            return TASK_DIR_MAP[task_key]
        raise ValueError(f"Unsupported task: {task}")

    def _load_metadata(self, meta_path: str) -> Optional[dict]:
        if not meta_path or not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_metadata_path(
        self,
        task: str,
        problem_name: Optional[str],
        metadata_path: Optional[str],
    ) -> str:
        _ = task, problem_name
        if not metadata_path:
            raise FileNotFoundError("metadata_path is required.")

        if metadata_path:
            if os.path.exists(metadata_path):
                return metadata_path
            candidate = os.path.join(self.project_root, metadata_path)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Metadata path not found: {metadata_path}")

    def _artifact_lookup(self, metadata: dict, key: str):
        artifacts = metadata.get("artifacts", {})
        if not isinstance(artifacts, dict):
            return None
        if key in artifacts:
            return artifacts[key]
        public = artifacts.get("public", {})
        if isinstance(public, dict) and key in public:
            return public[key]
        meta = artifacts.get("meta", {})
        if isinstance(meta, dict) and key in meta:
            return meta[key]
        debug = artifacts.get("debug", {})
        if isinstance(debug, dict) and key in debug:
            return debug[key]
        return None

    def _resolve_ref(self, base_dir: str, ref: str) -> str:
        if os.path.isabs(ref):
            return ref
        return os.path.join(base_dir, ref)

    def _find_csv_for_metadata(self, metadata: dict, meta_path: str) -> str:
        results_csv = self._artifact_lookup(metadata, "results_csv")
        if not results_csv:
            raise RuntimeError("Metadata missing artifacts.public.results_csv")
        candidate = self._resolve_ref(os.path.dirname(meta_path), results_csv)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"CSV not found: {candidate}")
        return candidate

    def _find_pkl_for_metadata(self, task: str, metadata: dict, meta_path: str) -> Optional[str]:
        if str(task).strip().upper() != "MODELER":
            return None
        model_ref = self._artifact_lookup(metadata, "model_path")
        if not model_ref:
            return None
        candidate = self._resolve_ref(os.path.dirname(meta_path), model_ref)
        if not os.path.exists(candidate):
            print(f"[ResultLoader] Model path not found: {candidate}")
            return None
        return candidate

    def load_task(
        self,
        *,
        task: str,
        problem_name: Optional[str] = None,
        csv_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        allow_latest_fallback: bool = False,
    ) -> TaskResult:
        if allow_latest_fallback:
            raise ValueError(
                "latest metadata fallback is no longer supported. "
                "Provide an explicit metadata_path."
            )

        meta_path = self._resolve_metadata_path(task, problem_name, metadata_path)
        metadata = self._load_metadata(meta_path)
        if not metadata:
            raise RuntimeError(f"Failed to load metadata: {meta_path}")
        if problem_name and metadata.get("problem") != problem_name:
            raise RuntimeError(
                "Problem mismatch between requested problem and metadata: "
                f"requested={problem_name}, metadata={metadata.get('problem')}"
            )

        if csv_path is None:
            csv_path = self._find_csv_for_metadata(metadata, meta_path)
        elif not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        pkl_path = self._find_pkl_for_metadata(task, metadata, meta_path)
        df = pd.read_csv(csv_path)

        return TaskResult(
            task=str(task).strip().upper(),
            problem_name=metadata.get("problem", problem_name or ""),
            csv_path=csv_path,
            pkl_path=pkl_path,
            df=df,
            metadata=metadata,
            metadata_path=meta_path,
        )
