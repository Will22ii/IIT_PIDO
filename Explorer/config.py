from dataclasses import dataclass, field

from CAE_tool_interface.config import CAEConfig


@dataclass
class ExplorerUserConfig:
    # 알려진 최적점 (시각화용, 옵션)
    known_optimum: object | None = None


@dataclass
class ExplorerSystemConfig:
    # 1) 샘플링 크기
    n_samples: int = 10000
    sample_multiplier: float | None = 20.0
    n_samples_min: int = 1000
    n_samples_max: int = 10000

    # 2) 경계 샘플링
    boundary_ratio: float = 0.2
    boundary_corner_ratio: float = 0.5

    # 3) bounds 확장/클리핑
    bounds_margin_ratio: float = 0.03
    bounds_min_volume_ratio: float = 0.249
    bounds_expansion_mode: str = "uniform"
    bounds_weight_clip_min: float = 0.7
    bounds_weight_clip_max: float = 1.5
    # EXP-3: 고제약(crate_hat < threshold) 시 fi_aware → uncertainty_aware 자동 스위치
    bounds_expansion_high_crate_switch_enabled: bool = True
    bounds_expansion_high_crate_threshold: float = 0.85

    # 4) 후보 선택
    quantile_threshold: float = 0.85
    min_topk_count: int = 30
    # DBSCAN 입력 후보 상한 (None이면 동적식 사용)
    max_topk_count: int | None = None
    max_topk_count_dynamic_enabled: bool = True
    max_topk_count_dynamic_scale: float = 60.0
    max_topk_count_dynamic_bias: float = 40.0
    max_topk_count_dynamic_min: int = 120
    max_topk_count_dynamic_max: int = 500

    # 5) 군집화
    dbscan_min_samples: int = 2
    dbscan_eps_quantile: float = 0.9

    # 6) post 제약 fallback
    post_lambda_default: float = 2.0

    # 7) 실행/메타
    save_plot: bool = True
    debug_level: str = "off"
    strategy_id: str = "S4_dual"
    probe_multistart: int = 20
    obj_refine_bounds_scale: float = 1.30
    strategy_params: dict[str, object] = field(default_factory=dict)


@dataclass
class ExplorerConfig:
    user: ExplorerUserConfig
    system: ExplorerSystemConfig
    cae: CAEConfig
    cae_metadata_path: str | None = None
    doe_csv_path: str | None = None
    doe_metadata_path: str | None = None
    model_pkl_path: str | None = None
    modeler_metadata_path: str | None = None
    fi_scores_path: str | None = None
