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
    # DOE objective + bounds 케이스에서 source mixture 사용 여부
    source_mixture_enabled: bool = True
    # source mixture 기본 확률 (topk / boundary / random)
    source_topk_prob: float = 0.60
    source_boundary_prob: float = 0.25
    source_random_prob: float = 0.15
    # focus3 budget class별 source mixture. n_opt / p 기준.
    focus3_budget_policy_enabled: bool = True
    focus3_budget_ultra_low_np: float = 3.0
    focus3_budget_low_np: float = 8.0
    focus3_budget_normal_np: float = 20.0
    focus3_ultra_low_source_topk_prob: float = 0.85
    focus3_ultra_low_source_boundary_prob: float = 0.10
    focus3_ultra_low_source_random_prob: float = 0.05
    focus3_low_source_topk_prob: float = 0.75
    focus3_low_source_boundary_prob: float = 0.15
    focus3_low_source_random_prob: float = 0.10
    focus3_normal_source_topk_prob: float = 0.60
    focus3_normal_source_boundary_prob: float = 0.25
    focus3_normal_source_random_prob: float = 0.15
    focus3_rich_source_topk_prob: float = 0.45
    focus3_rich_source_boundary_prob: float = 0.35
    focus3_rich_source_random_prob: float = 0.20
    # 정체(stagnation) 감지 시 탐색 강화
    source_stagnation_window: int = 8
    source_stagnation_tol: float = 1e-8
    source_stagnation_boundary_bonus: float = 0.10
    source_stagnation_random_bonus: float = 0.05
    # source pool 생성 파라미터
    source_pool_size: int = 24
    source_topk_fraction: float = 0.20
    source_topk_perturb_sigma: float = 0.08
    source_boundary_near_ratio: float = 0.03
    # pre-constraint 후보 생성 정책 (focus3 / point_converge)
    source_feasible_multiplier: int = 3
    source_feasible_retry: int = 3
    source_feasible_min_starts: int = 1
    # DOE 데이터에서 초기 학습점으로 사용할 상위 개수
    init_from_doe_topk: int = 20
    # DOE objective warm-start 상한 (이보다 많으면 top+diversity로 선별)
    init_max_points: int = 300
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
    # post feasibility penalty 사용 여부
    post_constraint_enabled: bool = True
    # score/acquisition 패널티 강도
    post_penalty_lambda: float = 2.0
    # p_feasible이 임계치보다 낮으면 추가 강한 패널티 부여
    post_p_feasible_min: float = 0.0
    post_p_feasible_hard_penalty: float = 0.0
    # post score 적용 방식: add_penalty(기본)
    post_score_mode: str = "add_penalty"
    # Deprecated: 실제 CAE objective 평가로 전환되어 현재 미사용
    surrogate_only_mode: bool = True
    # DOE가 없을 때 bootstrap 랜덤 샘플 수
    no_doe_bootstrap_size: int = 12
    # DOE가 없을 때 2-focus/3-focus 분기 임계값
    no_doe_mode_threshold: int = 200
    # DOE 없음 + two-focus 비율
    no_doe_stage1_ratio: float = 0.35
    # DOE 없음 + three-focus 비율
    no_doe_phase1_ratio: float = 0.25
    no_doe_phase2_ratio: float = 0.45
    # Deprecated: 실제 CAE objective 평가로 전환되어 현재 미사용
    no_doe_objective_proxy: str = "center_distance"
    # 중복 판정 반올림 자릿수
    dedup_decimals: int = 12
    # 디버그 산출물 생성 여부: off | on
    debug_level: str = "on"


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
