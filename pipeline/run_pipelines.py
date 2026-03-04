from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from DOE.config import DOEConfig, DOESystemConfig, DOEUserConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Modeler.config import ModelerConfig, ModelerSystemConfig, ModelerUserConfig
from pipeline.config import PipelineConfig, PipelineTasks
from pipeline.run_pipeline import run_pipeline


@dataclass(frozen=True)
class ProblemCase:
    problem_name: str
    known_optimum: Any
    n_samples: int
    objective_sense: str = "min"

PROBLEM_SUITE: list[ProblemCase] = [
    ProblemCase(
        problem_name="rosenbrock",
        known_optimum={"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0, "d1": 0.0},
        n_samples=1500,
    ),
    ProblemCase(
        problem_name="cantilever_beam",
        known_optimum={"H": 7.0, "h1": 0.1, "b1": 9.48482, "b2": 0.1, "d1": 0.0},
        n_samples=150,
    ),
    ProblemCase(
        problem_name="goldstein_price",
        known_optimum={"x1": 0.0, "x2": -1.0, "d1": 0.0},
        n_samples=500,
    ),
    ProblemCase(
        problem_name="six_hump_camel",
        known_optimum=[
            {"x1": 0.0898, "x2": -0.7126, "d1": 0.0},
            {"x1": -0.0898, "x2": 0.7126, "d1": 0.0},
        ],
        n_samples=50,
    ),
]


def _build_pipeline_config(
    *,
    case: ProblemCase,
    seed: int,
    run_doe: bool,
    run_modeler: bool,
    run_explorer: bool,
    use_additional: bool,
    use_hpo: bool,
    use_timestamp: bool,
) -> PipelineConfig:
    cae_cfg = CAEConfig(
        user=CAEUserConfig(
            problem_name=case.problem_name,
            seed=int(seed),
            objective_sense=str(case.objective_sense),
        ),
        system=CAESystemConfig(
            use_timestamp=bool(use_timestamp),
            allow_latest_fallback=False,
        ),
    )

    doe_cfg = DOEConfig(
        cae=cae_cfg,
        cae_user=None,
        user=DOEUserConfig(algo_name="lhs", use_additional=bool(use_additional)),
        system=DOESystemConfig(n_samples=int(case.n_samples)),
    )

    modeler_cfg = ModelerConfig(
        user=ModelerUserConfig(
            model_name="xgb",
            use_hpo=bool(use_hpo),
            use_secondary_selection=True,
        ),
        system=ModelerSystemConfig(),
        cae=cae_cfg,
        doe_csv_path=None,
        doe_metadata_path=None,
    )

    explorer_cfg = ExplorerConfig(
        user=ExplorerUserConfig(known_optimum=case.known_optimum),
        system=ExplorerSystemConfig(),
        cae=cae_cfg,
        doe_csv_path=None,
        doe_metadata_path=None,
        model_pkl_path=None,
        modeler_metadata_path=None,
    )

    return PipelineConfig(
        cae=cae_cfg,
        doe=doe_cfg if run_doe else None,
        modeler=modeler_cfg if run_modeler else None,
        explorer=explorer_cfg if run_explorer else None,
        tasks=PipelineTasks(
            run_doe=bool(run_doe),
            run_modeler=bool(run_modeler),
            run_explorer=bool(run_explorer),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner: execute multiple problems with repeated pipeline runs.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats for full problem suite.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--repeat-seed-step",
        type=int,
        default=1000,
        help="Seed offset added per repeat.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs even if one run fails.",
    )
    parser.add_argument("--no-additional", action="store_true", help="Disable DOE additional mode.")
    parser.add_argument("--no-hpo", action="store_true", help="Disable Modeler HPO.")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp in CAE/system config.")
    parser.add_argument("--skip-doe", action="store_true", help="Skip DOE stage.")
    parser.add_argument("--skip-modeler", action="store_true", help="Skip Modeler stage.")
    parser.add_argument("--skip-explorer", action="store_true", help="Skip Explorer stage.")
    args = parser.parse_args()

    repeats = max(int(args.repeats), 1)
    base_seed = int(args.base_seed)
    repeat_seed_step = int(args.repeat_seed_step)
    continue_on_error = bool(args.continue_on_error)

    run_doe = not bool(args.skip_doe)
    run_modeler = not bool(args.skip_modeler)
    run_explorer = not bool(args.skip_explorer)
    use_additional = not bool(args.no_additional)
    use_hpo = not bool(args.no_hpo)
    use_timestamp = not bool(args.no_timestamp)

    total_runs = repeats * len(PROBLEM_SUITE)
    run_counter = 0
    failures: list[dict[str, Any]] = []

    print("===================================")
    print(" Batch Pipeline 실행 시작")
    print("===================================")
    print(
        f"- repeats={repeats} total_runs={total_runs} "
        f"tasks(doe/modeler/explorer)={run_doe}/{run_modeler}/{run_explorer}"
    )

    for rep in range(repeats):
        for idx, case in enumerate(PROBLEM_SUITE):
            run_counter += 1
            seed = base_seed + rep * repeat_seed_step + idx
            print(
                "[Batch] "
                f"run={run_counter}/{total_runs} repeat={rep + 1}/{repeats} "
                f"problem={case.problem_name} seed={seed} n_samples={case.n_samples}"
            )

            cfg = _build_pipeline_config(
                case=case,
                seed=seed,
                run_doe=run_doe,
                run_modeler=run_modeler,
                run_explorer=run_explorer,
                use_additional=use_additional,
                use_hpo=use_hpo,
                use_timestamp=use_timestamp,
            )

            try:
                run_pipeline(config=cfg)
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "run": run_counter,
                    "repeat": rep + 1,
                    "problem": case.problem_name,
                    "seed": seed,
                    "error": str(exc),
                }
                failures.append(payload)
                print(
                    "[Batch] FAILED "
                    f"problem={case.problem_name} repeat={rep + 1} seed={seed} error={exc}"
                )
                if not continue_on_error:
                    raise

    print("===================================")
    print(" Batch Pipeline 실행 완료")
    print("===================================")
    print(f"- total_runs={total_runs}")
    print(f"- success_runs={total_runs - len(failures)}")
    print(f"- failed_runs={len(failures)}")
    if failures:
        print("- failure_details:")
        for item in failures:
            print(
                f"  run={item['run']} repeat={item['repeat']} "
                f"problem={item['problem']} seed={item['seed']} error={item['error']}"
            )


if __name__ == "__main__":
    main()
