import os
import json
from datetime import datetime
import pandas as pd


class ResultSaver:
    """
    Handle all result saving logic for a single execution(run).

    - CSV + metadata: stage-based (DOE / MODELER / ...)
    - TXT: workflow-based (final report)
    - timestamp: execution-level option
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

    def _get_stage_dir(self, stage: str) -> str:
        return os.path.join(self.project_root, "result", stage.lower())

    def save_stage_csv(
        self,
        *,
        stage: str,
        df: pd.DataFrame,
        problem_name: str,
        model_path: str | None = None,
        extra_metadata: dict | None = None,
    ) -> dict:
        output_dir = self._get_stage_dir(stage)
        os.makedirs(output_dir, exist_ok=True)

        csv_name = self._build_filename(
            base=f"{stage.lower()}_result_{problem_name}",
            ext=".csv",
        )
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False)

        meta_name = self._build_filename(
            base=f"{stage.lower()}_metadata_{problem_name}",
            ext=".json",
        )
        meta_path = os.path.join(output_dir, meta_name)

        metadata = {
            "stage": stage.upper(),
            "problem": problem_name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "created_at": datetime.now().isoformat(),
            "csv_path": csv_path,
        }

        if model_path:
            metadata["model_path"] = model_path

        if extra_metadata:
            metadata.update(extra_metadata)

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "csv": csv_path,
            "metadata": meta_path,
        }

    def save_stage_v2(
        self,
        *,
        run_root: str,
        stage: str,
        problem_name: str,
        df: pd.DataFrame,
        inputs: dict,
        resolved_params: dict,
        results: dict,
        artifacts: dict,
    ) -> dict:
        stage_dir = os.path.join(run_root, stage)
        artifacts_dir = os.path.join(stage_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        csv_name = f"{stage.lower()}_results.csv"
        csv_path = os.path.join(artifacts_dir, csv_name)
        df.to_csv(csv_path, index=False)

        metadata = {
            "stage": stage,
            "problem": problem_name,
            "created_at": datetime.now().isoformat(),
            "inputs": inputs,
            "resolved_params": resolved_params,
            "results": results,
            "artifacts": {
                "results_csv": os.path.relpath(csv_path, stage_dir),
                **artifacts,
            },
        }

        meta_path = os.path.join(stage_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "csv": csv_path,
            "metadata": meta_path,
            "stage_dir": stage_dir,
            "artifacts_dir": artifacts_dir,
        }

    def _build_result_directory(self, workflow_info: dict) -> str:
        base_result_dir = os.path.join(self.project_root, "result")

        active_stages = [
            key for key in ["DOE", "MODELER", "EXPLORER", "OPT"]
            if workflow_info.get(key) is not None
        ]

        if len(active_stages) == 1:
            return os.path.join(base_result_dir, active_stages[0].lower())

        return base_result_dir

    def _build_result_title(self, workflow_info: dict) -> str:
        stages = ["CAE"]
        for key in ["DOE", "MODELER", "EXPLORER", "OPT"]:
            if workflow_info.get(key) is not None:
                stages.append(key)
        return " + ".join(stages) + " RESULT"

    def save_final_txt(
        self,
        *,
        content_lines: list[str],
        problem_name: str,
        workflow_info: dict,
    ) -> str:
        output_dir = self._build_result_directory(workflow_info)
        os.makedirs(output_dir, exist_ok=True)

        filename = self._build_filename(
            base=f"result_{problem_name}",
            ext=".txt",
        )
        path = os.path.join(output_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write("========================================\n")
            f.write(f"{self._build_result_title(workflow_info)}\n")
            f.write("========================================\n\n")
            for line in content_lines:
                f.write(line + "\n")

        return path
