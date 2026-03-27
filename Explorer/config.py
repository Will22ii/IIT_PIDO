from dataclasses import dataclass, field

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
    boundary_corner_ratio: float = 0.5

    # 선택 경계 마진 기본 계수 (adaptive: m = base * (0.25 - raw_v) / 0.25, raw_v>=0.25면 off)
    bounds_margin_ratio: float = 0.03
    # 선택 경계 최소 부피비(역마진 floor). raw/final volume이 이 값보다 작으면 확장 시도
    bounds_min_volume_ratio: float = 0.25

    # 상위/하위 분위수 기준
    quantile_threshold: float = 0.90
    # 최소 top-k 보장 개수
    min_topk_count: int = 20

    # DBSCAN 설정값
    dbscan_min_samples: int = 2
    dbscan_eps_quantile: float = 0.9

    # Post 제약 결합 점수 fallback 람다 (DOE 메타에 post_lambda 없을 때 사용)
    post_lambda_default: float = 2.0

    # 시각화 옵션
    save_plot: bool = True
    # 출력 디버그 레벨: off | full
    debug_level: str = "off"
    # Explorer 전략 ID (배치 실험 구분용)
    strategy_id: str = "S4_dual"
    # Probe 기반 전략 시작점 개수(미사용 전략에서도 메타 기록용)
    probe_multistart: int = 20
    # 전략별 추가 파라미터 메타(실험 추적용)
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
