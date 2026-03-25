from dataclasses import dataclass

from CAE_tool_interface.config import CAEConfig, CAEUserConfig


@dataclass
class DOEUserConfig:
    # DOE 알고리즘 이름 (예: "lhs")
    algo_name: str = "lhs"
    # 추가 DOE 사용 여부
    use_additional: bool = False


@dataclass
class DOESystemConfig:
    # =========================
    # 1) DOE 기본
    # =========================
    n_samples: int = 120  # DOE 전체 샘플 수 (추가 DOE면 total_budget)
    force_baseline_initial: bool = False  # 초기 DOE에서 기준점 강제 포함 여부
    initial_corner_ratio: float = 0.05  # 초기 DOE에서 corner 샘플 비율

    # =========================
    # 2) 추가 DOE: 단계/비율
    # =========================
    additional_init_ratio: float = 0.4  # 초기 DOE 비율
    additional_exec_ratio: float = 0.1  # 단계별 실행점(X_exec) 비율
    additional_initial_probe_multiplier: float = 2.0  # 초기 DOE 제약 통과율 추정용 배수
    success_rate_floor: float = 0.02  # success 비율 하한 (미만이면 FAILED_SUCCESS_RATE_MIN)
    # NOTE:
    # global_random_ratio는 현재 전달/적용하지 않습니다.
    # 전역 X_exec은 boundary/margin/top 버킷을 우선 배정하고,
    # 남은 슬롯을 random으로 채우는 정책을 사용합니다.
    global_boundary_ratio: float = 0.2  # 전역 X_plan/X_exec 경계 샘플 비율
    global_margin_ratio: float = 0.2  # 전역 X_exec 중 제약 경계(margin) 버킷 비율
    global_top_ratio: float = 0.2  # 전역 X_exec 중 top-k 버킷 비율
    global_boundary_corner_ratio: float = 0.4  # 경계 샘플 중 코너 조합 비율

    # =========================
    # 3) 추가 DOE: X_plan 크기
    # =========================
    plan_base_k: float = 200.0  # n_plan 기본값 = plan_base_k * 차원수(dim)
    plan_remaining_cap: float = 4.0  # n_plan <= plan_remaining_cap * remaining
    plan_decay: float = 0.9  # 단계별 감쇠 비율
    plan_filter_safety: float = 1.2  # 제약 필터링 손실 보정 계수
    plan_filter_r_floor: float = 0.02  # 제약 통과(feasible) 비율 하한
    max_additional_stages: int = 12  # 추가 DOE 최대 단계 수

    # =========================
    # 4) Gate 설정
    # =========================
    # Gate1 (공간 안정성)
    gate1_ratio: float = 0.3  # Gate1 상위 k 비율 (X_plan 대비)
    gate1_pass_ratio: float = 0.6  # Gate1 소프트 자카드 통과 기준

    # Gate2 (불확실성)
    gate2_k: int = 2  # Gate2 카이제곱 자유도
    gate2_cdf_level: float = 0.9  # Gate2 CDF 수준
    gate2_ratio_threshold: float = 0.9  # Gate2 통과 비율 기준
    gate2_relax_factor: float = 1.1  # Gate2 임계값 완화 계수

    # =========================
    # 5) Local Planner: 앵커/클러스터
    # =========================
    # 앵커 총량은 stage별 max_base/decay로 관리
    local_anchor_max_base: int = 6  # 단계별 앵커 최대값 시작점
    local_anchor_max_decay: float = 0.9  # 단계별 앵커 최대값 감소율
    local_anchor_best_ratio: float = 0.35  # 상위 앵커 비율
    local_anchor_small_ratio: float = 0.2  # 소규모 군집 앵커 비율
    local_top_p: float = 0.3  # 앵커 후보 상위 p 비율
    local_dbscan_min_samples: int = 2  # 로컬 DBSCAN 최소 샘플 수
    local_dbscan_q_eps: float = 0.65  # 로컬 DBSCAN eps 분위수
    local_dbscan_eps_max: float = 0.25  # 로컬 DBSCAN eps 상한 (정규화 공간 기준)

    # =========================
    # 6) Local Planner: 샘플 스케일
    # =========================
    local_radius_ratio_phase1: float = 0.25  # 1단계 로컬 반경 비율
    local_radius_ratio_phase2: float = 0.18  # 2단계 로컬 반경 비율
    local_min_radius_ratio: float = 0.08  # 로컬 최소 반경 비율 (전역 span 기준)
    local_tol_ratio: float = 0.3  # 로컬 중복 제거 허용치 비율 (반경 평균 대비)

    # =========================
    # 7) Local Planner: Anchor Refiner(GP)
    # =========================
    local_refine_min_points: int = 15  # 로컬 GP refinement 최소 학습 개수
    local_cluster_delta_ratio: float = 0.03  # cluster box 확장폭 (전역 span 비율)
    local_singleton_box_ratio: float = 0.05  # singleton cluster box 폭 (전역 span 비율)
    local_phase1_kappa: float = 0.8  # phase1 LCB kappa
    local_phase2_kappa: float = 0.6  # phase2 LCB kappa
    local_base_perturb_ratio: float = 0.05  # base 주변 multistart perturb 비율
    local_gp_use_white_kernel: bool = False  # local/probe GP에서 WhiteKernel 사용 여부

    # =========================
    # 8) Local Planner: 제약 재시도
    # =========================
    local_constraint_retry_count: int = 1  # 로컬 제약 실패 시 반경 축소 재시도 횟수
    local_constraint_shrink_factor: float = 0.75  # 로컬 제약 실패 시 반경 축소 계수
    local_constraint_min_factor: float = 2.0  # f_i >= factor * (q_i + 1) 기준
    local_exec_pick_mode: str = "random"  # local X_exec 선택 방식: random | distance

    # =========================
    # 9) Post 제약 penalty
    # =========================
    post_use_penalty: bool = True  # post 제약 확률 기반 페널티 사용 여부
    post_lambda_init: float = 2.0  # post 페널티 람다 초기값
    post_lambda_min: float = 0.25  # post 페널티 람다 하한
    post_lambda_max: float = 8.0  # post 페널티 람다 상한
    post_lambda_power: float = 1.0  # 람다 적응식 지수
    post_feasible_rate_floor: float = 0.05  # post feasible 비율 하한(0 나눗셈 방지)
    post_clf_min_samples: int = 30  # post feasibility 분류모델 학습 최소 샘플 수
    post_clf_min_pos: int = 5  # post feasibility 분류모델 양성 최소 개수
    post_clf_min_neg: int = 5  # post feasibility 분류모델 음성 최소 개수

    # =========================
    # 10) Phase/종료 정책
    # =========================
    phase1_global_ratio: float = 0.8  # 1단계 전역 비율
    phase2_global_ratio: float = 0.35  # 2단계 전역 비율
    min_additional_rounds: int = 6  # 최소 추가 스테이지 횟수
    phase2_min_usable_np_ratio: float = 10.0  # phase1->2 전환 최소 usable N/p 비율 (strictly greater)
    stop_span_ratio_threshold: float = 0.20  # 수렴 종료 기준: span_ratio_mean
    stop_anchor_spread_streak: int = 5  # 수렴 종료 기준: anchor_spread_mean==0 연속 횟수
    stop_min_usable_np_ratio: float = 20.0  # 종료 허용 최소 usable N/p 비율 (usable: success && feasible)

    # =========================
    # 10-1) Probe Stage (전역 GP 1회 탐색)
    # =========================
    probe_stage_enabled: bool = True  # phase2 이력 + min_additional_rounds 이후 1회 실행
    probe_top_ratio: float = 0.3  # history 상위 비율(유망 영역 정의)
    probe_max_points: int = 6  # probe stage 최대 실행 점 수
    probe_min_range_ratio: float = 0.3  # 차원별 최소 탐색폭(전역 span 비율)
    probe_std_scale: float = 2.0  # std 기반 탐색폭 배수
    probe_perturb_ratio: float = 0.02  # best 주변 perturb 비율

    # =========================
    # 11) 기타
    # =========================
    additional_cfg: dict | None = None  # 임시 덮어쓰기 설정
    # 출력 디버그 레벨: off | full
    debug_level: str = "off"


@dataclass
class DOEConfig:
    # CAE 설정
    cae: CAEConfig
    # CAE 사용자 설정 (선택)
    cae_user: CAEUserConfig | None
    # DOE 사용자 설정
    user: DOEUserConfig
    # DOE 시스템 설정
    system: DOESystemConfig
    # CAE 출력 (선택)
    cae_output: dict | None = None
    # CAE 메타데이터 경로 (선택, run_context가 없을 때 필수)
    cae_metadata_path: str | None = None


 
