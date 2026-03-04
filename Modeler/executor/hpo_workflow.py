from dataclasses import dataclass
import hashlib
import json
import os
import sqlite3

import numpy as np
import pandas as pd

from Modeler.executor.hpo_runner import HPORunner


# -------------------------------------------------
# XGB HPO param type spec
# -------------------------------------------------
XGB_PARAM_TYPES = {
    "n_estimators": int,
    "max_depth": int,
    "min_child_weight": int,
    "subsample": float,
    "colsample_bytree": float,
    "learning_rate": float,
    "gamma": float,
}


@dataclass
class HPOResolveResult:
    best_params: dict | None
    hpo_params_used: bool
    hpo_signature: str | None
    data_hash: str | None


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_hpo_signature(
    *,
    problem_name: str,
    objective_sense: str,
    model_name: str,
    target_col: str,
    data_hash: str,
) -> str:
    payload = {
        "problem_name": problem_name,
        "objective_sense": objective_sense,
        "model_name": model_name,
        "target_col": target_col,
        "data_hash": data_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_hpo_params_by_signature(
    *,
    signature: str,
    project_root: str,
) -> dict | None:
    db_path = os.path.join(project_root, "result", "hpo_cache.sqlite3")
    if not os.path.exists(db_path):
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hpo_cache (
                    signature TEXT PRIMARY KEY,
                    params_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_ref TEXT
                )
                """
            )
            row = cur.execute(
                "SELECT params_json FROM hpo_cache WHERE signature = ?",
                (signature,),
            ).fetchone()
            if not row:
                return None
            payload = json.loads(row[0])
            return {
                k: XGB_PARAM_TYPES[k](v) if k in XGB_PARAM_TYPES else v
                for k, v in payload.items()
            }
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        return None


def update_hpo_cache(
    *,
    project_root: str,
    signature: str,
    params: dict,
    metadata_ref: str | None = None,
) -> None:
    result_root = os.path.join(project_root, "result")
    os.makedirs(result_root, exist_ok=True)
    db_path = os.path.join(result_root, "hpo_cache.sqlite3")
    now = pd.Timestamp.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hpo_cache (
                signature TEXT PRIMARY KEY,
                params_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_ref TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT INTO hpo_cache(signature, params_json, created_at, metadata_ref)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(signature) DO UPDATE SET
                params_json=excluded.params_json,
                created_at=excluded.created_at,
                metadata_ref=excluded.metadata_ref
            """,
            (signature, json.dumps(params, sort_keys=True), now, metadata_ref),
        )
        conn.commit()


def resolve_hpo_params(
    *,
    use_hpo: bool,
    model_name: str,
    hpo_config: dict | None,
    use_timestamp: bool,
    project_root: str,
    problem_name: str,
    objective_sense: str,
    target_col: str,
    csv_path: str | None,
    X: np.ndarray,
    y: np.ndarray,
    base_seed: int,
    kfold_splits: int,
) -> HPOResolveResult:
    best_params = None
    hpo_params_used = False
    hpo_signature = None
    data_hash = None

    if use_hpo:
        if not csv_path:
            raise RuntimeError("HPO requires resolved csv_path for data hash/signature.")
        data_hash = _file_sha256(csv_path)
        hpo_signature = _build_hpo_signature(
            problem_name=problem_name,
            objective_sense=objective_sense,
            model_name=model_name,
            target_col=target_col,
            data_hash=data_hash,
        )
        reuse_hpo = True
        if hpo_config is not None:
            reuse_hpo = bool(hpo_config.get("reuse_if_same_config", True))
        if reuse_hpo:
            reused = _load_hpo_params_by_signature(
                signature=hpo_signature,
                project_root=project_root,
            )
            if reused:
                best_params = reused
                hpo_params_used = True
                print("- Reusing HPO params (signature match)")

        if best_params is None:
            cfg = hpo_config or {}
            hpo_runner = HPORunner(
                n_trials=cfg.get("n_trials", 20),
                lambda_std=cfg.get("lambda_std", 0.5),
                use_timestamp=use_timestamp,
            )

            hpo_result = hpo_runner.run_xgb(
                X=X,
                y=y,
                base_random_seed=base_seed,
                problem_name=problem_name,
                kfold_splits=kfold_splits,
            )

            best_params = hpo_result["best_params"]
            hpo_params_used = True
            print("- HPO executed")

    elif model_name == "xgb":
        if not csv_path:
            raise RuntimeError("XGB cache reuse requires resolved csv_path for signature.")
        data_hash = _file_sha256(csv_path)
        hpo_signature = _build_hpo_signature(
            problem_name=problem_name,
            objective_sense=objective_sense,
            model_name=model_name,
            target_col=target_col,
            data_hash=data_hash,
        )
        reused = _load_hpo_params_by_signature(
            signature=hpo_signature,
            project_root=project_root,
        )
        if reused:
            best_params = reused
            hpo_params_used = True
            print("- Using cached HPO params (signature match)")
    else:
        best_params = None
        hpo_params_used = False

    return HPOResolveResult(
        best_params=best_params,
        hpo_params_used=hpo_params_used,
        hpo_signature=hpo_signature,
        data_hash=data_hash,
    )
