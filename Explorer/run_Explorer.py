from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Explorer.executor.explorer_orchestrator import ExplorerOrchestrator


def main():
    # Standalone 실행 시에는 metadata/model 경로를 명시하거나 pipeline run_context로 실행해야 한다.
    # Standalone example (custom data/model paths)
    # cfg = ExplorerConfig(
    #     user=ExplorerUserConfig(
    #         known_optimum={"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
    #     ),
    #     system=ExplorerSystemConfig(
    #         n_samples=10000,
    #         quantile_threshold=0.9,
    #         dbscan_min_samples=3,
    #         save_plot=True,
    #     ),
    #     cae=CAEConfig(
    #         user=CAEUserConfig(problem_name="goldstein_price", seed=42, objective_sense="min"),
    #         system=CAESystemConfig(use_timestamp=True, allow_latest_fallback=True),
    #     ),
    #     cae_metadata_path="result/run_<id>/CAE/metadata.json",
    #     doe_csv_path="result/run_<id>/DOE/artifacts/public/doe_results.csv",
    #     doe_metadata_path="result/run_<id>/DOE/metadata.json",
    #     model_pkl_path="result/run_<id>/Modeler/artifacts/public/modeler_selected_models.pkl",
    #     modeler_metadata_path="result/run_<id>/Modeler/metadata.json",
    # )
    cfg = ExplorerConfig(
        user=ExplorerUserConfig(
            known_optimum={"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
        ),
        system=ExplorerSystemConfig(
            n_samples=10000,
            quantile_threshold=0.9,
            dbscan_min_samples=3,
            save_plot=True,
        ),
        cae=CAEConfig(
            user=CAEUserConfig(problem_name="goldstein_price", seed=42, objective_sense="min"),
            system=CAESystemConfig(use_timestamp=True, allow_latest_fallback=False),
        ),
        cae_metadata_path=None,
        doe_csv_path=None,
        doe_metadata_path=None,
        model_pkl_path=None,
        modeler_metadata_path=None,
    )
    ExplorerOrchestrator(cfg).run()


if __name__ == "__main__":
    main()
