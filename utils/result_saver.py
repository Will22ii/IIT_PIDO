import json
import os
from datetime import datetime

import pandas as pd


TASK_DIR_MAP = {
    "CAE": "CAE",
    "DOE": "DOE",
    "MODELER": "Modeler",
    "EXPLORER": "Explorer",
    "OPT": "OPT",
}


class ResultSaver:
    """
    V3-only result saver.

    All task outputs are stored under one run root:
    - <run_root>/<TaskDir>/metadata.json
    - <run_root>/<TaskDir>/artifacts/public/*
    - <run_root>/<TaskDir>/artifacts/meta/*
    - <run_root>/<TaskDir>/artifacts/debug/*
    """

    def __init__(self, use_timestamp: bool = False):
        self.use_timestamp = use_timestamp
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    def _build_filename(self, base: str, ext: str) -> str:
        if self.use_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base}_{ts}{ext}"
        return f"{base}{ext}"

    def _task_dir_name(self, task: str) -> str:
        task_key = str(task).strip().upper()
        if task_key in TASK_DIR_MAP:
            return TASK_DIR_MAP[task_key]
        raise ValueError(f"Unsupported task: {task}")

    def _normalize_artifact_value(self, task_dir: str, value):
        if value is None:
            return None
        if isinstance(value, str):
            return os.path.relpath(value, task_dir) if os.path.isabs(value) else value
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    out.append(
                        os.path.relpath(item, task_dir) if os.path.isabs(item) else item
                    )
                else:
                    out.append(item)
            return out
        return value

    def _normalize_artifacts(self, task_dir: str, artifacts: dict | None) -> dict:
        if not artifacts:
            return {}
        normalized = {}
        for key, value in artifacts.items():
            normalized[key] = self._normalize_artifact_value(task_dir, value)
        return normalized

    def save_task_v3(
        self,
        *,
        run_root: str,
        task: str,
        problem_name: str,
        df: pd.DataFrame,
        inputs: dict,
        resolved_params: dict,
        results: dict,
        public_artifacts: dict | None = None,
        meta_artifacts: dict | None = None,
        debug_artifacts: dict | None = None,
    ) -> dict:
        task_key = str(task).strip().upper()
        task_dir = os.path.join(run_root, self._task_dir_name(task_key))
        artifacts_dir = os.path.join(task_dir, "artifacts")
        public_dir = os.path.join(artifacts_dir, "public")
        meta_dir = os.path.join(artifacts_dir, "meta")
        debug_dir = os.path.join(artifacts_dir, "debug")
        os.makedirs(public_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

        csv_name = f"{task_key.lower()}_results.csv"
        csv_path = os.path.join(public_dir, csv_name)
        df.to_csv(csv_path, index=False)

        public_refs = {
            "results_csv": os.path.relpath(csv_path, task_dir),
            **self._normalize_artifacts(task_dir, public_artifacts),
        }
        meta_refs = self._normalize_artifacts(task_dir, meta_artifacts)
        debug_refs = self._normalize_artifacts(task_dir, debug_artifacts)

        artifacts = {
            "public": public_refs,
            "meta": meta_refs,
            "debug": debug_refs,
        }

        # Convenience aliases for internal/debug tools that read direct keys.
        for layer_name in ("public", "meta", "debug"):
            for key, value in artifacts[layer_name].items():
                if key not in artifacts:
                    artifacts[key] = value

        metadata = {
            "schema_version": "3.0",
            "task": task_key,
            "problem": problem_name,
            "created_at": datetime.now().isoformat(),
            "inputs": inputs or {},
            "resolved_params": resolved_params or {},
            "results": results or {},
            "artifacts": artifacts,
        }

        meta_path = os.path.join(task_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "csv": csv_path,
            "metadata": meta_path,
            "task_dir": task_dir,
            "artifacts_dir": artifacts_dir,
            "public_dir": public_dir,
            "meta_dir": meta_dir,
            "debug_dir": debug_dir,
        }

    def _build_result_title(self, workflow_info: dict) -> str:
        stages = ["CAE"]
        for key in ["DOE", "MODELER", "EXPLORER", "OPT"]:
            if workflow_info.get(key) is not None:
                stages.append(key)
        return " + ".join(stages) + " RESULT"

    def save_final_txt(
        self,
        *,
        run_root: str,
        task: str,
        content_lines: list[str],
        problem_name: str,
        workflow_info: dict,
    ) -> str:
        task_dir = os.path.join(run_root, self._task_dir_name(task))
        public_dir = os.path.join(task_dir, "artifacts", "public")
        os.makedirs(public_dir, exist_ok=True)
        filename = self._build_filename(
            base=f"result_{problem_name}",
            ext=".txt",
        )
        path = os.path.join(public_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write("========================================\n")
            f.write(f"{self._build_result_title(workflow_info)}\n")
            f.write("========================================\n\n")
            for line in content_lines:
                f.write(line + "\n")

        return path
