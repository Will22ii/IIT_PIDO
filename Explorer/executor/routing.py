"""routed_v2: dual 모드 내부 라우팅 (rule-based, interpretable).

`dse/SKILL.md`의 phase_gate에 따라 단일 dual 전략으로 수렴하기 위한 결정 트리.
모든 결정은 pre-hoc 관측 가능한 state feature만으로 유도되며,
사후 라벨(`survivor_optimum_included`, `joint_pass`, `fail_type`,
`volume_cap_pass`, `volume_ratio`)은 입력으로 사용하지 않는다.

호출 컨텍스트:
  Explorer dual 분기에서 `dual_policy_mode == "routed_v2"`일 때 호출.
  반환된 RouterOutput은 obj_ratio, tilt_strength, dse2_iou_threshold,
  cap_target, expansion_mode 등에 적용되며 reason_trace는 analysis meta로 dump.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AcqMode = Literal["ei", "lcb", "union"]
ExpansionMode = Literal["fi_aware", "uncertainty_aware"]


def _clip(v: float, lo: float, hi: float) -> float:
    if v != v:  # NaN
        return lo
    return float(min(max(v, lo), hi))


@dataclass
class RouterInput:
    """routed_v2 입력 state. 모두 pre-hoc, observable.

    None 허용 필드는 해당 정보가 미가용일 때 보수적 default로 결정된다.
    """
    # topology
    p_dim: int
    has_pre_constraints: bool
    has_post_constraints: bool = False

    # data density
    usable_n: int = 0
    usable_n_over_p: float = 0.0
    constraint_rate_hat: float = 1.0

    # DOE quality
    doe_gate1_pass_rate: float | None = None
    doe_gate2_pass_rate: float | None = None
    doe_gate1_score_last: float | None = None
    doe_gate2_score_last: float | None = None
    phase2_entered: bool = False
    phase2_transition_count: int = 0
    doe_collapse_detected_count: int = 0
    doe_diversity_injection_applied_count: int = 0

    # pred quality (Explorer interim)
    pred_cluster_confidence: float | None = None
    pred_cluster_sigma_mean_selected: float | None = None
    pred_refine_shift_norm: float | None = None

    # disagreement (computed inline if not provided)
    pred_obj_iou: float | None = None
    dual_disagreement_iou: float | None = None
    dual_disagreement_center_l1_norm: float | None = None


@dataclass
class RouterOutput:
    """routed_v2 결정. 각 필드는 orchestrator의 해당 변수를 override한다.

    `reason_trace`는 결정 근거 로그(필수). analysis meta에 dump되어 추후
    shadow validation 및 router accuracy 분석에 사용된다.
    """
    acq_mode: AcqMode
    obj_ratio: float
    tilt_strength: float
    expansion_mode: ExpansionMode
    cap_target: float
    dse2_iou_threshold: float
    reason_trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "acq_mode": self.acq_mode,
            "obj_ratio": float(self.obj_ratio),
            "tilt_strength": float(self.tilt_strength),
            "expansion_mode": self.expansion_mode,
            "cap_target": float(self.cap_target),
            "dse2_iou_threshold": float(self.dse2_iou_threshold),
            "reason_trace": list(self.reason_trace),
        }


# ─────────────────────────────────────────────────────────────────
# 결정 트리 임계값 (모두 명시적 — 변경 시 reason_trace에도 반영됨)
# ─────────────────────────────────────────────────────────────────
THRESH = {
    # Acquisition mode
    "acq_ei_pred_conf_min": 0.55,
    "acq_ei_g2_pass_min": 0.80,
    "acq_ei_np_min": 18.0,
    "acq_lcb_pred_conf_max": 0.40,
    "acq_lcb_g2_pass_max": 0.50,
    "acq_lcb_np_max": 12.0,

    # obj_ratio base
    "obj_base_constraint": 0.70,
    "obj_base_low_dim_dense_p_max": 3,
    "obj_base_low_dim_dense_np_min": 20.0,
    "obj_base_low_dim_dense": 0.85,
    "obj_base_high_dim_dense_p_min": 5,
    "obj_base_high_dim_dense_np_min": 30.0,
    "obj_base_high_dim_dense": 0.45,
    "obj_base_default": 0.55,

    # obj_ratio adjustments
    "obj_iou_strong_disag": 0.20,
    "obj_iou_strong_disag_bonus": 0.15,
    "obj_iou_mild_disag": 0.30,
    "obj_iou_mild_disag_bonus": 0.08,
    "obj_high_crate_threshold": 0.85,
    "obj_high_crate_bonus": 0.10,
    "obj_ratio_min": 0.25,
    "obj_ratio_max": 0.85,

    # tilt
    "tilt_high_crate_threshold": 0.85,
    "tilt_high_crate": 0.55,
    "tilt_severe_disag_iou": 0.15,
    "tilt_severe_disag": 0.55,
    "tilt_default": 0.45,
    "tilt_min": 0.30,
    "tilt_max": 0.65,

    # expansion mode (state-based; 기존 EXP-3 로직과 정합)
    "exp_high_crate_threshold": 0.85,

    # fixed cap target (hard policy)
    "cap_fixed": 0.2499,

    # DSE-2 trigger
    "dse2_high_conf_threshold": 0.55,
    "dse2_iou_high_conf": 0.18,
    "dse2_iou_default": 0.22,
    # DSE-2 adaptive relax (uncertainty-aware)
    "dse2_low_conf_threshold": 0.58,
    "dse2_low_conf_bonus": 0.02,
    "dse2_low_g2_pass_threshold": 0.60,
    "dse2_low_g2_pass_bonus": 0.01,
    "dse2_shift_high_threshold": 0.035,
    "dse2_shift_high_bonus": 0.02,
    "dse2_disagree_center_high_threshold": 0.10,
    "dse2_disagree_center_high_bonus": 0.02,
    "dse2_low_np_threshold": 14.0,
    "dse2_low_np_bonus": 0.01,
    "dse2_iou_min": 0.16,
    "dse2_iou_max": 0.26,
}


def route_v2(
    state: RouterInput,
    overrides: dict | None = None,
) -> RouterOutput:
    """state → RouterOutput. overrides로 일부 임계 in-test 조정 가능.

    Args:
      state: pre-hoc observable state.
      overrides: THRESH 키별 부분 override (테스트/실험용). None이면 default.

    Returns:
      RouterOutput with full reason_trace.
    """
    T = dict(THRESH)
    if overrides:
        T.update(overrides)
    R: list[str] = []

    p = int(max(state.p_dim, 1))
    np_r = float(state.usable_n_over_p) if state.usable_n_over_p else 0.0
    has_c = bool(state.has_pre_constraints)
    crate = float(state.constraint_rate_hat) if state.constraint_rate_hat is not None else 1.0
    pc = state.pred_cluster_confidence
    g1p = state.doe_gate1_pass_rate
    g2p = state.doe_gate2_pass_rate
    iou = state.dual_disagreement_iou
    if iou is None:
        iou = state.pred_obj_iou  # fallback (동일 계산식)

    # ─────────────────────────────────────────────────────────────
    # (A) acquisition mode
    # 근거: shared_snapshot에서 EI(S4) vs LCB(S8)가 benchmark별로 ±2pt 수준.
    #       단일 acq 선택은 risky → 확신 영역에서만 단독, 나머지는 union.
    # ─────────────────────────────────────────────────────────────
    if (
        pc is not None
        and pc >= T["acq_ei_pred_conf_min"]
        and g2p is not None
        and g2p >= T["acq_ei_g2_pass_min"]
        and np_r >= T["acq_ei_np_min"]
    ):
        acq_mode: AcqMode = "ei"
        R.append(
            f"acq=ei: pred_conf={pc:.3f}>={T['acq_ei_pred_conf_min']} & "
            f"g2_pass={g2p:.3f}>={T['acq_ei_g2_pass_min']} & "
            f"np={np_r:.1f}>={T['acq_ei_np_min']}"
        )
    elif (
        (pc is not None and pc < T["acq_lcb_pred_conf_max"])
        or (g2p is not None and g2p < T["acq_lcb_g2_pass_max"])
        or (np_r < T["acq_lcb_np_max"])
    ):
        acq_mode = "lcb"
        R.append(
            f"acq=lcb: surrogate uncertain "
            f"(pred_conf={pc}, g2_pass={g2p}, np={np_r:.1f})"
        )
    else:
        acq_mode = "union"
        R.append(
            f"acq=union: mixed (pred_conf={pc}, g2_pass={g2p}, np={np_r:.1f}) "
            f"→ EI ∪ LCB bounds"
        )

    # ─────────────────────────────────────────────────────────────
    # (B) obj_ratio (pred vs obj 비율)
    # 근거: routed_v1 베이스 + DSE-2 분석 (low IoU에서 obj 우세 효과)
    # ─────────────────────────────────────────────────────────────
    if has_c:
        base = T["obj_base_constraint"]
        R.append(f"obj_ratio base={base:.2f} (has_pre_constraints)")
    elif p <= T["obj_base_low_dim_dense_p_max"] and np_r > T["obj_base_low_dim_dense_np_min"]:
        base = T["obj_base_low_dim_dense"]
        R.append(f"obj_ratio base={base:.2f} (low-dim p={p}, dense np={np_r:.1f})")
    elif p >= T["obj_base_high_dim_dense_p_min"] and np_r > T["obj_base_high_dim_dense_np_min"]:
        base = T["obj_base_high_dim_dense"]
        R.append(f"obj_ratio base={base:.2f} (high-dim p={p}, dense np={np_r:.1f})")
    else:
        base = T["obj_base_default"]
        R.append(f"obj_ratio base={base:.2f} (mid)")

    if iou is not None and iou < T["obj_iou_strong_disag"]:
        base += T["obj_iou_strong_disag_bonus"]
        R.append(
            f"obj_ratio +{T['obj_iou_strong_disag_bonus']:.2f} "
            f"(iou={iou:.4f}<{T['obj_iou_strong_disag']:.2f} strong disagreement)"
        )
    elif iou is not None and iou < T["obj_iou_mild_disag"]:
        base += T["obj_iou_mild_disag_bonus"]
        R.append(
            f"obj_ratio +{T['obj_iou_mild_disag_bonus']:.2f} "
            f"(iou={iou:.4f}<{T['obj_iou_mild_disag']:.2f} mild disagreement)"
        )

    if has_c and crate < T["obj_high_crate_threshold"]:
        base += T["obj_high_crate_bonus"]
        R.append(
            f"obj_ratio +{T['obj_high_crate_bonus']:.2f} "
            f"(crate_hat={crate:.3f}<{T['obj_high_crate_threshold']:.2f})"
        )

    obj_ratio = _clip(base, T["obj_ratio_min"], T["obj_ratio_max"])

    # ─────────────────────────────────────────────────────────────
    # (C) tilt_strength
    # 근거: cantilever pass tilt 평균 0.434 vs fail 0.388 → tilt↑ 도움
    # ─────────────────────────────────────────────────────────────
    if has_c and crate < T["tilt_high_crate_threshold"]:
        tilt = T["tilt_high_crate"]
        R.append(
            f"tilt={tilt:.2f} (high crate: crate_hat={crate:.3f})"
        )
    elif iou is not None and iou < T["tilt_severe_disag_iou"]:
        tilt = T["tilt_severe_disag"]
        R.append(
            f"tilt={tilt:.2f} (severe disag: iou={iou:.4f}<{T['tilt_severe_disag_iou']:.2f})"
        )
    else:
        tilt = T["tilt_default"]
        R.append(f"tilt={tilt:.2f} (default)")

    tilt = _clip(tilt, T["tilt_min"], T["tilt_max"])

    # ─────────────────────────────────────────────────────────────
    # (D) expansion_mode
    # 근거: 기존 EXP-3 로직 (high-crate 시 fi_aware → uncertainty_aware)
    # ─────────────────────────────────────────────────────────────
    if has_c and crate < T["exp_high_crate_threshold"]:
        expansion_mode: ExpansionMode = "uncertainty_aware"
        R.append(
            f"expansion=uncertainty_aware (high crate: crate_hat={crate:.3f})"
        )
    else:
        expansion_mode = "fi_aware"
        R.append("expansion=fi_aware (default; no FI score 시 uniform fallback)")

    # ─────────────────────────────────────────────────────────────
    # (E) cap_target (hard fixed policy)
    # 30% 완화 경로를 제거하고 고정 cap(24.99%)만 사용한다.
    # ─────────────────────────────────────────────────────────────
    cap_target = T["cap_fixed"]
    R.append("cap=0.2499 (fixed hard cap; relaxed path removed)")

    # ─────────────────────────────────────────────────────────────
    # (F) DSE-2 trigger threshold
    # pred_conf 높을수록 conservative (덜 자주 union)
    # uncertainty 신호(low conf / low gate2 / high shift / high center disagreement
    # / low n/p)에서는 threshold를 완화해 union 빈도를 높인다.
    # ─────────────────────────────────────────────────────────────
    base_dse2_iou: float
    if pc is not None and pc >= T["dse2_high_conf_threshold"]:
        base_dse2_iou = float(T["dse2_iou_high_conf"])
        R.append(
            f"dse2_iou base={base_dse2_iou:.2f} (pred_conf={pc:.3f}>="
            f"{T['dse2_high_conf_threshold']:.2f} → narrower trigger)"
        )
    else:
        base_dse2_iou = float(T["dse2_iou_default"])
        R.append(f"dse2_iou base={base_dse2_iou:.2f} (default/low pred conf)")

    dse2_relax = 0.0
    relax_reasons: list[str] = []
    if pc is None or float(pc) < float(T["dse2_low_conf_threshold"]):
        _bonus = float(T["dse2_low_conf_bonus"])
        dse2_relax += _bonus
        relax_reasons.append(f"low_conf(+{_bonus:.2f})")
    if g2p is not None and float(g2p) < float(T["dse2_low_g2_pass_threshold"]):
        _bonus = float(T["dse2_low_g2_pass_bonus"])
        dse2_relax += _bonus
        relax_reasons.append(f"low_g2(+{_bonus:.2f})")
    if (
        state.pred_refine_shift_norm is not None
        and float(state.pred_refine_shift_norm) >= float(T["dse2_shift_high_threshold"])
    ):
        _bonus = float(T["dse2_shift_high_bonus"])
        dse2_relax += _bonus
        relax_reasons.append(f"high_shift(+{_bonus:.2f})")
    if (
        state.dual_disagreement_center_l1_norm is not None
        and float(state.dual_disagreement_center_l1_norm)
        >= float(T["dse2_disagree_center_high_threshold"])
    ):
        _bonus = float(T["dse2_disagree_center_high_bonus"])
        dse2_relax += _bonus
        relax_reasons.append(f"high_center_disag(+{_bonus:.2f})")
    if np_r < float(T["dse2_low_np_threshold"]):
        _bonus = float(T["dse2_low_np_bonus"])
        dse2_relax += _bonus
        relax_reasons.append(f"low_np(+{_bonus:.2f})")

    dse2_iou = _clip(
        float(base_dse2_iou + dse2_relax),
        float(T["dse2_iou_min"]),
        float(T["dse2_iou_max"]),
    )
    if relax_reasons:
        R.append(
            f"dse2_iou relax: base={base_dse2_iou:.2f} +{dse2_relax:.2f} "
            f"→ {dse2_iou:.2f} ({', '.join(relax_reasons)})"
        )
    else:
        R.append(f"dse2_iou final={dse2_iou:.2f} (no uncertainty relax)")

    return RouterOutput(
        acq_mode=acq_mode,
        obj_ratio=float(obj_ratio),
        tilt_strength=float(tilt),
        expansion_mode=expansion_mode,
        cap_target=float(cap_target),
        dse2_iou_threshold=float(dse2_iou),
        reason_trace=R,
    )


__all__ = ["RouterInput", "RouterOutput", "route_v2", "THRESH"]
