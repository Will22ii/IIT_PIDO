from CAE_tool_interface.config import CAEConfig, CAEUserConfig, CAESystemConfig
from Explorer.config import ExplorerConfig, ExplorerSystemConfig, ExplorerUserConfig
from Explorer.executor.explorer_orchestrator import ExplorerOrchestrator


def main():
    # Development default: use latest metadata when explicit paths are not provided.
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
    #     doe_csv_path="C:\\python\\project\\result\\doe\\doe_result_goldstein_price.csv",
    #     doe_metadata_path="C:\\python\\project\\result\\doe\\doe_metadata_goldstein_price.json",
    #     model_pkl_path="C:\\python\\project\\result\\modeler\\modeler_selected_models_goldstein_price.pkl",
    #     modeler_metadata_path="C:\\python\\project\\result\\modeler\\modeler_metadata_goldstein_price_summary.json",
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
            system=CAESystemConfig(use_timestamp=True, allow_latest_fallback=True),
        ),
        doe_csv_path=None,
        doe_metadata_path=None,
        model_pkl_path=None,
        modeler_metadata_path=None,
    )
    ExplorerOrchestrator(cfg).run()


if __name__ == "__main__":
    main()
