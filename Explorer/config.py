from dataclasses import dataclass

from CAE_tool_interface.config import CAEConfig


@dataclass
class ExplorerUserConfig:
    # 알려진 최적점 (시각화용, 옵션)
    known_optimum: object | None = None


@dataclass
class ExplorerSystemConfig:
    # 샘플 수 (기본 고정값)
    n_samples: int = 5000
    # DOE 데이터 수 × 배수로 n_samples 계산 (값이 없으면 고정값 사용)
    sample_multiplier: float | None = 4.0
    # 배수 계산 사용 시 최소/최대 제한
    n_samples_min: int = 500
    n_samples_max: int = 10000

    # 경계 샘플 비율 (0~1)
    boundary_ratio: float = 0.1
    # 경계 샘플 중 코너 조합 비율 (0~1)
    boundary_corner_ratio: float = 0.4

    # 선택 경계 마진 비율 (평균 span 기준, 0이면 미사용)
    bounds_margin_ratio: float = 0.0

    # 상위/하위 분위수 기준
    quantile_threshold: float = 0.95
    # 최소 top-k 보장 개수
    min_topk_count: int = 20

    # DBSCAN 설정값
    dbscan_min_samples: int = 2
    dbscan_eps_quantile: float = 0.9

    # Post 제약 결합 점수 fallback 람다 (DOE 메타에 post_lambda 없을 때 사용)
    post_lambda_default: float = 2.0

    # 시각화 옵션
    save_plot: bool = True
    tsne_max_points: int = 2000
    # 출력 디버그 레벨: off | full
    debug_level: str = "off"


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
