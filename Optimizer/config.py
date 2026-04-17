from dataclasses import dataclass

from CAE_tool_interface.config import CAEConfig


@dataclass
class OptimizerUserConfig:
    # Optimizer 반복 횟수(=새 점 탐색 횟수)
    n_samples: int = 30
    # Standalone 입력: DOE 결과 CSV 경로 (옵션)
    doe_csv_path: str | None = None
    # Standalone 입력: Explorer selected bounds JSON 경로 (옵션)
    explorer_bounds_path: str | None = None


@dataclass
class OptimizerSystemConfig:
    # acquisition 정책: auto | LCB | EI
    acq_type: str = "auto"
    # LCB kappa 스케줄 (auto/LCB에서 사용)
    kappa_start: float = 2.5
    kappa_end: float = 0.6
    # EI 파라미터
    ei_xi: float = 0.01
    # acquisition 다중 시작점 최적화 파라미터
    n_restarts: int = 8
    starts_per_iter: int = 32
    random_starts_ratio: float = 0.7
    # DOE 데이터에서 초기 학습점으로 사용할 상위 개수
    init_from_doe_topk: int = 20
    # DOE seed 사용 범위: in_bounds | all
    doe_seed_scope: str = "in_bounds"
    # GP 재학습 주기 (iteration 단위)
    gp_refit_every: int = 1
    # DOE objective 컬럼명
    objective_col: str = "objective"
    # None이면 CAE metadata objective_sense 사용
    objective_sense_override: str | None = None
    # 현재는 기본 OFF (추후 pre/post 제약 로직 확장용)
    enforce_pre_constraints: bool = False
    # benchmark 함수 미연결 상태에서 surrogate 예측값으로 loop 진행
    surrogate_only_mode: bool = True
    # DOE가 없을 때 bootstrap 랜덤 샘플 수
    no_doe_bootstrap_size: int = 12
    # DOE가 없을 때 2-stage/3-phase 분기 임계값
    no_doe_mode_threshold: int = 200
    # DOE 없음 + two-stage 비율
    no_doe_stage1_ratio: float = 0.35
    # DOE 없음 + three-phase 비율
    no_doe_phase1_ratio: float = 0.25
    no_doe_phase2_ratio: float = 0.45
    # DOE 없음 + objective proxy (surrogate_only_mode=True일 때)
    # center_distance | random
    no_doe_objective_proxy: str = "center_distance"
    # 중복 판정 반올림 자릿수
    dedup_decimals: int = 12
    # 로그 레벨: off | full
    debug_level: str = "off"


@dataclass
class OptimizerConfig:
    user: OptimizerUserConfig
    system: OptimizerSystemConfig
    cae: CAEConfig
    # sequential 연계용 metadata 경로들 (옵션)
    cae_metadata_path: str | None = None
    doe_metadata_path: str | None = None
    explorer_metadata_path: str | None = None
    modeler_metadata_path: str | None = None
    # 직접 경로 주입 (옵션)
    doe_csv_path: str | None = None
    explorer_bounds_path: str | None = None
