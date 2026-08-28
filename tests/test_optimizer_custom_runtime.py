from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from CAE_tool_interface.config import CAEConfig, CAESystemConfig, CAEUserConfig
from Optimizer.algorithms.registry import (
    describe_optimizer_algorithms,
    discover_custom_optimizer_algorithms,
    get_optimizer_algorithm,
    list_optimizer_algorithm_errors,
)
from Optimizer.config import (
    OptimizerConfig,
    OptimizerSystemConfig,
    OptimizerSystemConfigView,
    OptimizerUserConfig,
    split_optimizer_system_config,
    optimizer_system_config_view,
)
from Optimizer.run_Optimizer import run_optimizer
from pipeline.run_context import RunContext


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_DIR = os.path.join(PROJECT_ROOT, "Optimizer", "algorithms", "custom")
CUSTOM_FILE = os.path.join(CUSTOM_DIR, "zz_test_runtime_contract.py")
INVALID_CUSTOM_FILE = os.path.join(CUSTOM_DIR, "zz_test_invalid_contract.py")
ALGORITHM_ID = "test_runtime_contract"


CUSTOM_ALGORITHM_SOURCE = f'''
ALGORITHM_ID = "{ALGORITHM_ID}"


class Algorithm:
    def run(self, *, runtime, config, resolved):
        points = runtime.algorithm_params.get("points") or [
            {{"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0}}
        ]
        for point in points:
            if runtime.should_stop:
                break
            runtime.evaluate(
                point,
                source_mode=ALGORITHM_ID,
                segment="custom",
            )
        return runtime.build_result(algorithm_id=ALGORITHM_ID)
'''


def _write_test_custom_algorithm() -> None:
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    with open(CUSTOM_FILE, "w", encoding="utf-8") as f:
        f.write(CUSTOM_ALGORITHM_SOURCE)
    discover_custom_optimizer_algorithms(force=True)


def _remove_test_custom_algorithm() -> None:
    if os.path.exists(CUSTOM_FILE):
        os.remove(CUSTOM_FILE)
    if os.path.exists(INVALID_CUSTOM_FILE):
        os.remove(INVALID_CUSTOM_FILE)
    discover_custom_optimizer_algorithms(force=True)


def _build_temp_run_context(tmpdir: str, constraint_defs: list[dict] | None = None) -> RunContext:
    user_config_path = os.path.join(tmpdir, "user_config_snapshot.json")
    with open(user_config_path, "w", encoding="utf-8") as f:
        json.dump({"problem": "rosenbrock_nodummy", "task": "OPT"}, f)

    index_path = os.path.join(tmpdir, "index.json")
    cae_dir = os.path.join(tmpdir, "CAE")
    os.makedirs(cae_dir, exist_ok=True)
    cae_metadata_path = os.path.join(cae_dir, "metadata.json")
    variables = [
        {"name": f"x{i}", "lb": -2.048, "ub": 2.048}
        for i in range(1, 6)
    ]
    with open(cae_metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "3.0",
                "task": "CAE",
                "problem": "rosenbrock_nodummy",
                "inputs": {
                    "variables": variables,
                    "constraint_defs": list(constraint_defs or []),
                    "user_config": "../user_config_snapshot.json",
                },
                "resolved_params": {
                    "seed": 123,
                    "objective_sense": "min",
                },
                "results": {},
                "artifacts": {},
            },
            f,
            indent=2,
        )
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "3.0",
                "run_id": "optimizer_custom_runtime_test",
                "tasks": {"CAE": os.path.join("CAE", "metadata.json")},
            },
            f,
            indent=2,
        )
    return RunContext(
        run_id="optimizer_custom_runtime_test",
        run_root=tmpdir,
        user_config_snapshot_path=user_config_path,
        index_path=index_path,
    )


class OptimizerCustomRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        _write_test_custom_algorithm()

    def tearDown(self) -> None:
        _remove_test_custom_algorithm()

    def test_custom_algorithm_discovery_reports_runtime_api(self) -> None:
        description = describe_optimizer_algorithms()
        algorithms = {
            item["algorithm_id"]: item
            for item in description["algorithms"]
        }

        self.assertIn(ALGORITHM_ID, algorithms)
        self.assertEqual(algorithms[ALGORITHM_ID]["source"], "custom")
        self.assertTrue(algorithms[ALGORITHM_ID]["uses_runtime_api"])
        self.assertEqual(get_optimizer_algorithm(ALGORITHM_ID).__class__.__name__, "Algorithm")

    def test_custom_algorithm_discovery_reports_bad_files_without_breaking_registry(self) -> None:
        with open(INVALID_CUSTOM_FILE, "w", encoding="utf-8") as f:
            f.write(
                'ALGORITHM_ID = "bad_contract"\n\n'
                "class Algorithm:\n"
                "    def run(self, *, config, resolved):\n"
                "        return None\n"
            )

        errors = discover_custom_optimizer_algorithms(force=True)
        description = describe_optimizer_algorithms()
        algorithms = {item["algorithm_id"] for item in description["algorithms"]}

        self.assertIn(os.path.basename(INVALID_CUSTOM_FILE), errors)
        self.assertIn(os.path.basename(INVALID_CUSTOM_FILE), list_optimizer_algorithm_errors())
        self.assertIn("runtime", errors[os.path.basename(INVALID_CUSTOM_FILE)])
        self.assertIn("focus_bo", algorithms)
        self.assertIn(ALGORITHM_ID, algorithms)
        self.assertEqual(get_optimizer_algorithm("focus_bo").__class__.__name__, "FocusBOAlgorithm")

    def test_optimizer_system_config_split_marks_common_and_algorithm_knobs(self) -> None:
        focus_system = OptimizerSystemConfig(algorithm_id="focus_bo")
        focus_view = optimizer_system_config_view(focus_system, algorithm_id="focus_bo")
        focus_payload = split_optimizer_system_config(focus_system, algorithm_id="focus_bo")

        self.assertIsInstance(focus_view, OptimizerSystemConfigView)
        self.assertEqual(focus_payload["algorithm_contract"], "focus_bo_builtin")
        self.assertEqual(focus_payload["common"]["algorithm_id"], "focus_bo")
        self.assertIn("objective_col", focus_payload["common"])
        self.assertIn("focus_pipeline", focus_payload["algorithm"])
        self.assertIn("focus3_refine_starts", focus_payload["algorithm"])

        custom_system = OptimizerSystemConfig(
            algorithm_id=ALGORITHM_ID,
            algorithm_params={"alpha": 0.25},
        )
        custom_payload = split_optimizer_system_config(custom_system, algorithm_id=ALGORITHM_ID)

        self.assertEqual(custom_payload["algorithm_contract"], "optimizer_runtime")
        self.assertEqual(custom_payload["common"]["algorithm_id"], ALGORITHM_ID)
        self.assertEqual(custom_payload["algorithm"], {"algorithm_params": {"alpha": 0.25}})
        self.assertIn("focus_pipeline", custom_payload["focus_bo"])

    def test_custom_runtime_outputs_are_algorithm_neutral(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="optimizer_custom_runtime_")
        try:
            run_context = _build_temp_run_context(tmpdir)
            config = OptimizerConfig(
                user=OptimizerUserConfig(n_samples=1, goal=1.2),
                system=OptimizerSystemConfig(
                    algorithm_id=ALGORITHM_ID,
                    debug_level="off",
                ),
                cae=CAEConfig(
                    user=CAEUserConfig(
                        problem_name="rosenbrock_nodummy",
                        seed=123,
                        objective_sense="min",
                    ),
                    system=CAESystemConfig(use_timestamp=False),
                ),
            )

            out = run_optimizer(config=config, run_context=run_context)

            self.assertAlmostEqual(out["best_objective_raw"], 1.0)
            best_point_path = os.path.join(
                tmpdir,
                "OPT",
                "artifacts",
                "public",
                "best_point.json",
            )
            metadata_path = os.path.join(tmpdir, "OPT", "metadata.json")
            with open(best_point_path, "r", encoding="utf-8") as f:
                best_payload = json.load(f)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["schema_version"], "3.0")
            self.assertEqual(metadata["task"], "OPT")
            self.assertEqual(best_payload["result_status"], "optimized")
            self.assertEqual(best_payload["algorithm_id"], ALGORITHM_ID)
            self.assertEqual(best_payload["algorithm_engine"], "optimizer_runtime")
            self.assertFalse(best_payload["focus3_executed"])
            self.assertEqual(
                metadata["resolved_params"]["optimizer_output_schema"],
                "optimizer_points_v2",
            )
            self.assertEqual(
                metadata["resolved_params"]["optimizer_progress_scheme"],
                "custom_algorithm",
            )
            self.assertIn("optimizer_algorithm", metadata["artifacts"]["meta"])
            self.assertIn("optimizer_system_config", metadata["artifacts"]["meta"])
            self.assertNotIn("focus_regions", metadata["artifacts"]["meta"])
            self.assertNotIn("focus_bounds", metadata["artifacts"]["meta"])
            self.assertIn("best_point", metadata["artifacts"]["public"])
            self.assertIn("optimizer_points", metadata["artifacts"]["public"])
            optimizer_system_config_path = os.path.join(
                os.path.dirname(metadata_path),
                metadata["artifacts"]["meta"]["optimizer_system_config"],
            )
            with open(optimizer_system_config_path, "r", encoding="utf-8") as f:
                system_payload = json.load(f)
            self.assertEqual(system_payload["schema"], "optimizer_system_config_v1")
            self.assertEqual(system_payload["algorithm_id"], ALGORITHM_ID)
            self.assertEqual(system_payload["algorithm_contract"], "optimizer_runtime")
            self.assertEqual(system_payload["common"]["algorithm_id"], ALGORITHM_ID)
            self.assertEqual(system_payload["algorithm"], {"algorithm_params": {}})
            self.assertIn("focus_pipeline", system_payload["focus_bo"])

            optimizer_algorithm_path = os.path.join(
                os.path.dirname(metadata_path),
                metadata["artifacts"]["meta"]["optimizer_algorithm"],
            )
            with open(optimizer_algorithm_path, "r", encoding="utf-8") as f:
                algorithm_payload = json.load(f)
            self.assertEqual(algorithm_payload["algorithm_id"], ALGORITHM_ID)
            self.assertEqual(algorithm_payload["algorithm_kind"], "runtime")
            self.assertEqual(algorithm_payload["schema"]["optimizer_output_schema"], "optimizer_points_v2")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_custom_runtime_pre_equality_penalty_keeps_public_objective_raw(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="optimizer_custom_runtime_preeq_")
        try:
            run_context = _build_temp_run_context(
                tmpdir,
                constraint_defs=[
                    {
                        "id": "x1_eq_zero",
                        "scope": "pre",
                        "type": "==",
                        "expr": "x1",
                        "limit": 0.0,
                        "eps": 0.0,
                    }
                ],
            )
            config = OptimizerConfig(
                user=OptimizerUserConfig(n_samples=2),
                system=OptimizerSystemConfig(
                    algorithm_id=ALGORITHM_ID,
                    debug_level="off",
                    pre_eq_penalty_lambda=100.0,
                    algorithm_params={
                        "points": [
                            {"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
                            {"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0, "x5": 0.0},
                        ],
                    },
                ),
                cae=CAEConfig(
                    user=CAEUserConfig(
                        problem_name="rosenbrock_nodummy",
                        seed=123,
                        objective_sense="min",
                    ),
                    system=CAESystemConfig(use_timestamp=False),
                ),
            )

            out = run_optimizer(config=config, run_context=run_context)

            self.assertAlmostEqual(out["best_point"]["x1"], 0.0)
            self.assertAlmostEqual(out["best_objective"], 5.0)
            self.assertAlmostEqual(out["best_point_raw"]["x1"], 1.0)
            self.assertAlmostEqual(out["best_objective_raw"], 1.0)

            best_point_path = os.path.join(
                tmpdir,
                "OPT",
                "artifacts",
                "public",
                "best_point.json",
            )
            with open(best_point_path, "r", encoding="utf-8") as f:
                best_payload = json.load(f)

            self.assertAlmostEqual(best_payload["best_objective"], 5.0)
            self.assertAlmostEqual(best_payload["best_objective_raw"], 1.0)
            self.assertTrue(best_payload["final_pre_best_feasible"])
            self.assertEqual(best_payload["final_pre_best_status"], "ok")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_custom_runtime_marks_no_final_pre_feasible_best_as_exploratory(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="optimizer_custom_runtime_no_preeq_")
        try:
            run_context = _build_temp_run_context(
                tmpdir,
                constraint_defs=[
                    {
                        "id": "x1_eq_zero",
                        "scope": "pre",
                        "type": "==",
                        "expr": "x1",
                        "limit": 0.0,
                        "eps": 0.0,
                    }
                ],
            )
            config = OptimizerConfig(
                user=OptimizerUserConfig(n_samples=1),
                system=OptimizerSystemConfig(
                    algorithm_id=ALGORITHM_ID,
                    debug_level="off",
                    pre_eq_penalty_lambda=100.0,
                    algorithm_params={
                        "points": [
                            {"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
                        ],
                    },
                ),
                cae=CAEConfig(
                    user=CAEUserConfig(
                        problem_name="rosenbrock_nodummy",
                        seed=123,
                        objective_sense="min",
                    ),
                    system=CAESystemConfig(use_timestamp=False),
                ),
            )

            run_optimizer(config=config, run_context=run_context)

            best_point_path = os.path.join(
                tmpdir,
                "OPT",
                "artifacts",
                "public",
                "best_point.json",
            )
            with open(best_point_path, "r", encoding="utf-8") as f:
                best_payload = json.load(f)

            self.assertEqual(best_payload["result_status"], "exploratory_best")
            self.assertEqual(best_payload["optimizer_status_basis"], "no_hard_pre_feasible_best")
            self.assertEqual(best_payload["best_point"], {})
            self.assertFalse(best_payload["final_pre_best_feasible"])
            self.assertEqual(best_payload["final_pre_best_status"], "no_pre_feasible_candidate")
            self.assertAlmostEqual(best_payload["best_effort_point"]["x1"], 1.0)
            self.assertAlmostEqual(best_payload["best_effort_objective"], 1.0)
            self.assertGreater(best_payload["best_effort_pre_violation_score"], 0.0)
            self.assertAlmostEqual(best_payload["least_violation_point"]["x1"], 1.0)
            self.assertAlmostEqual(best_payload["raw_best_infeasible_point"]["x1"], 1.0)
            self.assertGreaterEqual(len(best_payload["best_effort_pareto_points"]), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_focus_bo_outputs_keep_focus_artifacts(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="optimizer_focus_bo_")
        try:
            run_context = _build_temp_run_context(tmpdir)
            config = OptimizerConfig(
                user=OptimizerUserConfig(n_samples=1),
                system=OptimizerSystemConfig(
                    algorithm_id="focus_bo",
                    debug_level="on",
                ),
                cae=CAEConfig(
                    user=CAEUserConfig(
                        problem_name="rosenbrock_nodummy",
                        seed=123,
                        objective_sense="min",
                    ),
                    system=CAESystemConfig(use_timestamp=False),
                ),
            )

            run_optimizer(config=config, run_context=run_context)

            metadata_path = os.path.join(tmpdir, "OPT", "metadata.json")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["schema_version"], "3.0")
            self.assertEqual(metadata["task"], "OPT")
            self.assertEqual(
                metadata["resolved_params"]["optimizer_progress_scheme"],
                "focus",
            )
            self.assertEqual(
                metadata["resolved_params"]["optimizer_output_schema"],
                "optimizer_points_v2_focus",
            )
            self.assertIn("optimizer_algorithm", metadata["artifacts"]["meta"])
            self.assertIn("optimizer_system_config", metadata["artifacts"]["meta"])
            self.assertIn("focus_regions", metadata["artifacts"]["meta"])
            self.assertIn("focus_bounds", metadata["artifacts"]["meta"])
            self.assertIn("focus2_bounds_evolution_csv", metadata["artifacts"]["debug"])
            self.assertIn("focus3_trajectory_csv", metadata["artifacts"]["debug"])

            system_config_path = os.path.join(
                os.path.dirname(metadata_path),
                metadata["artifacts"]["meta"]["optimizer_system_config"],
            )
            with open(system_config_path, "r", encoding="utf-8") as f:
                system_payload = json.load(f)
            self.assertEqual(system_payload["algorithm_contract"], "focus_bo_builtin")
            self.assertIn("focus_pipeline", system_payload["algorithm"])
            self.assertIn("focus3_refine_starts", system_payload["algorithm"])

            optimizer_algorithm_path = os.path.join(
                os.path.dirname(metadata_path),
                metadata["artifacts"]["meta"]["optimizer_algorithm"],
            )
            with open(optimizer_algorithm_path, "r", encoding="utf-8") as f:
                algorithm_payload = json.load(f)
            self.assertEqual(algorithm_payload["algorithm_id"], "focus_bo")
            self.assertEqual(algorithm_payload["algorithm_kind"], "focus")
            self.assertEqual(
                algorithm_payload["schema"]["optimizer_output_schema"],
                "optimizer_points_v2_focus",
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
