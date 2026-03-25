import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict


@dataclass
class FeatureSelectionConfig:
    perm_min_pass_rate: float = 0.6
    perm_epsilon: float = 0.04
    use_score_drop: bool = True
    drop_metric: str = "drop_sq"
    drop_min_pass_rate: float = 0.6
    drop_epsilon: float = 0.06
    # very_low_data 전용 drop 채널 임계값
    drop_min_pass_rate_very_low_data: float = 0.35
    drop_epsilon_very_low_data: float = 0.02
    # fold vote weights
    weight_abs: float = 0.75
    weight_quantile: float = 0.15
    weight_rank: float = 0.10
    # channel merge weights
    weight_perm: float = 0.75
    weight_drop: float = 0.25
    # low_data 전용 채널 가중치 (low_data=True일 때 weight_perm/weight_drop 대신 사용)
    weight_perm_low_data: float = 0.90
    weight_drop_low_data: float = 0.10
    # scale merge default weights
    weight_global_default: float = 0.6
    # scale merge adaptive settings
    weight_global_low: float = 0.6
    weight_global_rich: float = 0.5
    elite_small_threshold: int = 40
    elite_rich_threshold: int = 80
    elite_mode: str = "bonus"
    elite_bonus_beta: float = 0.30
    # decision guards
    final_score_threshold: float = 0.6
    global_score_floor: float = 0.2
    # stability gate
    stability_enabled: bool = True
    stability_rule: str = "or"   # legacy fallback
    # 3단계 stability 분기
    stability_very_low_data_n_threshold: int = 50
    stability_rule_very_low_data: str = "or"
    stability_perm_min_rate_very_low_data: float = 0.55
    stability_drop_min_rate_very_low_data: float = 0.35
    stability_rule_low_data: str = "and"
    stability_perm_min_rate_low_data: float = 0.60
    stability_drop_min_rate_low_data: float = 0.44
    stability_rule_normal: str = "or"
    stability_perm_min_rate_normal: float = 0.80
    stability_drop_min_rate_normal: float = 0.60
    # disagreement penalty
    disagreement_penalty_enabled: bool = True
    disagreement_threshold: float = 0.25
    disagreement_penalty_scale: float = 0.40
    # null-importance soft gate
    null_enabled: bool = True
    null_mode: str = "soft"
    null_quantile: float = 0.95
    null_shuffle_runs_low_data: int = 25
    null_shuffle_runs_normal: int = 30
    null_alpha_low_data: float = 0.15
    null_alpha_normal: float = 0.12
    null_apply_to: str = "global_score"
    # quantile policy
    quantile_top_ratio_default: float = 0.30
    quantile_top_ratio_p_le_6: float = 0.50
    quantile_top_ratio_p_le_12: float = 0.40
    quantile_top_ratio_p_gt_12: float = 0.30


class FeatureSelector:
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config

    def run(
        self,
        *,
        perm_effect_path: str | None = None,
        problem_name: str,
        score_drop_path: str | None = None,
        perm_effect_elite_path: str | None = None,
        score_drop_elite_path: str | None = None,
        perm_effect_df: pd.DataFrame | None = None,
        perm_effect_elite_df: pd.DataFrame | None = None,
        score_drop_df: pd.DataFrame | None = None,
        score_drop_elite_df: pd.DataFrame | None = None,
        low_data: bool = False,
        n_features: int | None = None,
        n_elite: int | None = None,
        n_samples: int | None = None,
    ) -> Dict[str, pd.DataFrame]:
        _ = problem_name

        # perm global: DataFrame 우선, 없으면 path에서 읽기
        if perm_effect_df is not None and not perm_effect_df.empty:
            perm_global_raw = perm_effect_df
        elif perm_effect_path:
            perm_global_raw = pd.read_csv(perm_effect_path)
        else:
            perm_global_raw = pd.DataFrame()

        # perm elite
        if perm_effect_elite_df is not None and not perm_effect_elite_df.empty:
            perm_elite_raw = perm_effect_elite_df
        elif perm_effect_elite_path and self._has_rows(perm_effect_elite_path):
            perm_elite_raw = pd.read_csv(perm_effect_elite_path)
        else:
            perm_elite_raw = pd.DataFrame()

        # drop global
        drop_global_raw = pd.DataFrame()
        if self.config.use_score_drop:
            if score_drop_df is not None and not score_drop_df.empty:
                drop_global_raw = score_drop_df
            elif score_drop_path and self._has_rows(score_drop_path):
                drop_global_raw = pd.read_csv(score_drop_path)

        # drop elite
        drop_elite_raw = pd.DataFrame()
        if self.config.use_score_drop:
            if score_drop_elite_df is not None and not score_drop_elite_df.empty:
                drop_elite_raw = score_drop_elite_df
            elif score_drop_elite_path and self._has_rows(score_drop_elite_path):
                drop_elite_raw = pd.read_csv(score_drop_elite_path)

        p_dim = int(n_features) if n_features is not None else self._infer_n_features(
            perm_global_raw=perm_global_raw,
            drop_global_raw=drop_global_raw,
        )
        top_ratio = self._resolve_top_ratio(
            low_data=bool(low_data),
            p_dim=max(p_dim, 1),
        )

        very_low_n_thr = int(getattr(self.config, "stability_very_low_data_n_threshold", 50))
        is_very_low_data = bool(low_data) and int(n_samples or 0) < very_low_n_thr
        eff_drop_epsilon = (
            float(getattr(self.config, "drop_epsilon_very_low_data", self.config.drop_epsilon))
            if is_very_low_data else float(self.config.drop_epsilon)
        )
        eff_drop_min_pass_rate = (
            float(getattr(self.config, "drop_min_pass_rate_very_low_data", self.config.drop_min_pass_rate))
            if is_very_low_data else float(self.config.drop_min_pass_rate)
        )

        perm_proc_g, perm_sum_g = self._score_channel(
            raw=perm_global_raw,
            metric_src_col="delta",
            metric_name="perm_delta",
            epsilon=float(self.config.perm_epsilon),
            min_pass_rate=float(self.config.perm_min_pass_rate),
            top_ratio=float(top_ratio),
            scale_label="global",
        )
        perm_proc_e, perm_sum_e = self._score_channel(
            raw=perm_elite_raw,
            metric_src_col="delta",
            metric_name="perm_delta",
            epsilon=float(self.config.perm_epsilon),
            min_pass_rate=float(self.config.perm_min_pass_rate),
            top_ratio=float(top_ratio),
            scale_label="elite",
        )

        drop_proc_g, drop_sum_g = self._score_drop_channel(
            raw=drop_global_raw,
            epsilon=eff_drop_epsilon,
            min_pass_rate=eff_drop_min_pass_rate,
            top_ratio=float(top_ratio),
            scale_label="global",
        )
        drop_proc_e, drop_sum_e = self._score_drop_channel(
            raw=drop_elite_raw,
            epsilon=eff_drop_epsilon,
            min_pass_rate=eff_drop_min_pass_rate,
            top_ratio=float(top_ratio),
            scale_label="elite",
        )

        score_g = self._merge_scale_scores(
            perm_summary=perm_sum_g,
            drop_summary=drop_sum_g,
            scale_label="global",
            low_data=bool(low_data),
        )
        score_e = self._merge_scale_scores(
            perm_summary=perm_sum_e,
            drop_summary=drop_sum_e,
            scale_label="elite",
            low_data=bool(low_data),
        )

        selected = self._finalize_selection(
            global_df=score_g,
            elite_df=score_e,
            perm_processed_global=perm_proc_g,
            drop_processed_global=drop_proc_g,
            n_elite=(int(n_elite) if n_elite is not None else 0),
            low_data=bool(low_data),
            n_samples=(int(n_samples) if n_samples is not None else 0),
        )

        perm_processed = pd.concat([perm_proc_g, perm_proc_e], ignore_index=True)
        drop_processed = pd.concat([drop_proc_g, drop_proc_e], ignore_index=True)
        return {
            "importance_processed_pred": perm_processed,
            "importance_summary_pred_global": perm_sum_g,
            "importance_summary_pred_elite": perm_sum_e,
            "importance_processed_drop": drop_processed,
            "importance_summary_drop_global": drop_sum_g,
            "importance_summary_drop_elite": drop_sum_e,
            "selected_features": selected,
        }

    @staticmethod
    def _has_rows(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                next(f)
                return True
        except Exception:
            return False

    @staticmethod
    def _infer_n_features(*, perm_global_raw: pd.DataFrame, drop_global_raw: pd.DataFrame) -> int:
        for df in (perm_global_raw, drop_global_raw):
            if df is not None and not df.empty and "feature" in df.columns:
                return int(df["feature"].nunique())
        return 1

    def _resolve_top_ratio(self, *, low_data: bool, p_dim: int) -> float:
        if not low_data:
            return float(np.clip(self.config.quantile_top_ratio_default, 0.05, 1.0))
        if p_dim <= 6:
            return float(np.clip(self.config.quantile_top_ratio_p_le_6, 0.05, 1.0))
        if p_dim <= 12:
            return float(np.clip(self.config.quantile_top_ratio_p_le_12, 0.05, 1.0))
        return float(np.clip(self.config.quantile_top_ratio_p_gt_12, 0.05, 1.0))

    def _normalize_vote_weights(self) -> tuple[float, float, float]:
        w_abs = float(self.config.weight_abs)
        w_q = float(self.config.weight_quantile)
        w_r = float(self.config.weight_rank)
        total = w_abs + w_q + w_r
        if total <= 0.0:
            return 1.0, 0.0, 0.0
        return w_abs / total, w_q / total, w_r / total

    @staticmethod
    def _add_fold_norm_and_ranking(
        *,
        fold_df: pd.DataFrame,
        metric_col: str,
        epsilon: float,
        top_ratio: float,
    ) -> pd.DataFrame:
        out = fold_df.copy()
        out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce").fillna(0.0)
        fold_max = out.groupby("fold")[metric_col].transform("max").replace(0.0, np.nan)
        out["metric_norm"] = out[metric_col] / fold_max
        out["metric_norm"] = out["metric_norm"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out["abs_pass"] = out["metric_norm"] >= float(epsilon)
        q_level = float(1.0 - np.clip(top_ratio, 0.0, 1.0))
        q_thr = out.groupby("fold")["metric_norm"].transform(lambda s: s.quantile(q_level))
        out["quantile_pass"] = out["metric_norm"] >= q_thr

        rank = out.groupby("fold")["metric_norm"].rank(method="average", ascending=False)
        p_fold = out.groupby("fold")["feature"].transform("count").astype(float)
        denom = np.maximum(p_fold - 1.0, 1.0)
        out["rank_score"] = np.where(
            p_fold <= 1.0,
            1.0,
            (p_fold - rank) / denom,
        )
        out["rank_score"] = out["rank_score"].clip(0.0, 1.0)
        return out

    def _score_channel(
        self,
        *,
        raw: pd.DataFrame,
        metric_src_col: str,
        metric_name: str,
        epsilon: float,
        min_pass_rate: float,
        top_ratio: float,
        scale_label: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if raw is None or raw.empty:
            return pd.DataFrame(), pd.DataFrame()
        if metric_src_col not in raw.columns:
            return pd.DataFrame(), pd.DataFrame()

        fold_df = raw.copy()
        fold_df = fold_df.rename(columns={metric_src_col: metric_name})
        if "scale" not in fold_df.columns:
            fold_df["scale"] = str(scale_label)
        fold_df = self._add_fold_norm_and_ranking(
            fold_df=fold_df,
            metric_col=metric_name,
            epsilon=float(epsilon),
            top_ratio=float(top_ratio),
        )
        w_abs, w_q, w_r = self._normalize_vote_weights()
        fold_df["vote"] = (
            w_abs * fold_df["abs_pass"].astype(float)
            + w_q * fold_df["quantile_pass"].astype(float)
            + w_r * fold_df["rank_score"].astype(float)
        )

        stats = (
            fold_df.groupby("feature")[metric_name]
            .agg(metric_mean="mean", metric_min="min", metric_max="max")
            .reset_index()
        )
        global_max = float(stats["metric_mean"].max()) if len(stats) > 0 else 0.0
        global_max = global_max if global_max > 0 else 1.0
        stats["metric_mean_norm"] = stats["metric_mean"] / global_max

        summary = (
            fold_df.groupby("feature")
            .agg(
                abs_selection_rate=("abs_pass", "mean"),
                quantile_selection_rate=("quantile_pass", "mean"),
                rank_score_mean=("rank_score", "mean"),
                vote_score=("vote", "mean"),
            )
            .reset_index()
            .merge(stats, on="feature", how="left")
        )
        summary["epsilon"] = float(epsilon)
        summary["selected"] = summary["vote_score"] >= float(min_pass_rate)
        summary["reason"] = np.where(summary["selected"], "vote_pass", "vote_fail")
        summary["scale"] = str(scale_label)

        processed = fold_df[
            [
                "feature",
                "fold",
                "scale",
                metric_name,
                "metric_norm",
                "abs_pass",
                "quantile_pass",
                "rank_score",
                "vote",
            ]
        ].copy()
        return processed, summary

    def _score_drop_channel(
        self,
        *,
        raw: pd.DataFrame,
        epsilon: float,
        min_pass_rate: float,
        top_ratio: float,
        scale_label: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if raw is None or raw.empty or not self.config.use_score_drop:
            return pd.DataFrame(), pd.DataFrame()
        metric = str(self.config.drop_metric).strip().lower()
        if metric not in {"drop", "drop_sq"}:
            metric = "drop_sq"
        if metric not in raw.columns:
            return pd.DataFrame(), pd.DataFrame()
        fold_df = raw.copy()
        if "scale" not in fold_df.columns:
            fold_df["scale"] = str(scale_label)
        fold_df["drop_metric"] = pd.to_numeric(fold_df[metric], errors="coerce").fillna(0.0)
        fold_df["drop_metric"] = fold_df["drop_metric"].clip(lower=0.0)

        out = self._add_fold_norm_and_ranking(
            fold_df=fold_df.rename(columns={"drop_metric": "drop_metric_raw"}),
            metric_col="drop_metric_raw",
            epsilon=float(epsilon),
            top_ratio=float(top_ratio),
        )
        out = out.rename(columns={"drop_metric_raw": "drop_metric"})
        w_abs, w_q, w_r = self._normalize_vote_weights()
        out["vote"] = (
            w_abs * out["abs_pass"].astype(float)
            + w_q * out["quantile_pass"].astype(float)
            + w_r * out["rank_score"].astype(float)
        )

        stats = (
            out.groupby("feature")["drop_metric"]
            .agg(metric_mean="mean", metric_min="min", metric_max="max")
            .reset_index()
        )
        global_max = float(stats["metric_mean"].max()) if len(stats) > 0 else 0.0
        global_max = global_max if global_max > 0 else 1.0
        stats["metric_mean_norm"] = stats["metric_mean"] / global_max

        summary = (
            out.groupby("feature")
            .agg(
                abs_selection_rate=("abs_pass", "mean"),
                quantile_selection_rate=("quantile_pass", "mean"),
                rank_score_mean=("rank_score", "mean"),
                vote_score=("vote", "mean"),
            )
            .reset_index()
            .merge(stats, on="feature", how="left")
        )
        summary["epsilon"] = float(epsilon)
        summary["selected"] = summary["vote_score"] >= float(min_pass_rate)
        summary["reason"] = np.where(summary["selected"], "vote_pass", "vote_fail")
        summary["drop_metric_name"] = metric
        summary["scale"] = str(scale_label)

        keep_cols = [
            "feature",
            "fold",
            "scale",
            "r2_base",
            "r2_perm",
            "drop",
            "drop_sq",
            "drop_metric",
            "metric_norm",
            "abs_pass",
            "quantile_pass",
            "rank_score",
            "vote",
        ]
        keep_cols = [c for c in keep_cols if c in out.columns]
        processed = out[keep_cols].copy()
        return processed, summary

    def _merge_scale_scores(
        self,
        *,
        perm_summary: pd.DataFrame,
        drop_summary: pd.DataFrame,
        scale_label: str,
        low_data: bool = False,
    ) -> pd.DataFrame:
        if perm_summary is None or perm_summary.empty:
            return pd.DataFrame()

        base = perm_summary.copy().rename(
            columns={
                "abs_selection_rate": "perm_abs_selection_rate",
                "quantile_selection_rate": "perm_quantile_selection_rate",
                "rank_score_mean": "perm_rank_score_mean",
                "vote_score": "perm_vote_score",
                "metric_mean": "delta_mean",
                "metric_min": "delta_min",
                "metric_max": "delta_max",
                "metric_mean_norm": "delta_mean_norm",
                "epsilon": "perm_epsilon",
                "selected": "perm_selected",
                "reason": "perm_reason",
            }
        )

        if self.config.use_score_drop and drop_summary is not None and not drop_summary.empty:
            drop = drop_summary.copy().rename(
                columns={
                    "abs_selection_rate": "drop_abs_selection_rate",
                    "quantile_selection_rate": "drop_quantile_selection_rate",
                    "rank_score_mean": "drop_rank_score_mean",
                    "vote_score": "drop_vote_score",
                    "metric_mean": "drop_metric_mean",
                    "metric_min": "drop_metric_min",
                    "metric_max": "drop_metric_max",
                    "metric_mean_norm": "drop_metric_mean_norm",
                    "epsilon": "drop_epsilon",
                    "selected": "drop_selected",
                    "reason": "drop_reason",
                }
            )
            out = base.merge(
                drop[
                    [
                        "feature",
                        "drop_abs_selection_rate",
                        "drop_quantile_selection_rate",
                        "drop_rank_score_mean",
                        "drop_vote_score",
                        "drop_metric_mean",
                        "drop_metric_min",
                        "drop_metric_max",
                        "drop_metric_mean_norm",
                        "drop_metric_name",
                        "drop_epsilon",
                        "drop_selected",
                        "drop_reason",
                    ]
                ],
                on="feature",
                how="left",
            )
            out["drop_vote_score"] = pd.to_numeric(out["drop_vote_score"], errors="coerce").fillna(0.0)
        else:
            out = base.copy()
            out["drop_abs_selection_rate"] = 0.0
            out["drop_quantile_selection_rate"] = 0.0
            out["drop_rank_score_mean"] = 0.0
            out["drop_vote_score"] = 0.0
            out["drop_metric_mean"] = 0.0
            out["drop_metric_min"] = 0.0
            out["drop_metric_max"] = 0.0
            out["drop_metric_mean_norm"] = 0.0
            out["drop_metric_name"] = str(self.config.drop_metric).strip().lower()
            out["drop_epsilon"] = float(self.config.drop_epsilon)
            out["drop_selected"] = not bool(self.config.use_score_drop)
            out["drop_reason"] = "drop_disabled"

        if bool(low_data):
            w_perm = float(getattr(self.config, "weight_perm_low_data", self.config.weight_perm))
            w_drop = float(getattr(self.config, "weight_drop_low_data", self.config.weight_drop)) if bool(self.config.use_score_drop) else 0.0
        else:
            w_perm = float(self.config.weight_perm)
            w_drop = float(self.config.weight_drop) if bool(self.config.use_score_drop) else 0.0
        denom = w_perm + w_drop
        if denom <= 0.0:
            w_perm_n, w_drop_n = 1.0, 0.0
        else:
            w_perm_n, w_drop_n = w_perm / denom, w_drop / denom
        out["scale_score"] = (
            w_perm_n * pd.to_numeric(out["perm_vote_score"], errors="coerce").fillna(0.0)
            + w_drop_n * pd.to_numeric(out["drop_vote_score"], errors="coerce").fillna(0.0)
        )
        out["scale"] = str(scale_label)
        return out

    def _resolve_scale_weights(self, *, low_data: bool, n_elite: int) -> tuple[float, float]:
        if bool(low_data) or int(n_elite) < int(self.config.elite_small_threshold):
            wg = float(self.config.weight_global_low)
        elif int(n_elite) >= int(self.config.elite_rich_threshold):
            wg = float(self.config.weight_global_rich)
        else:
            wg = float(self.config.weight_global_default)
        we = 1.0 - wg
        return float(np.clip(wg, 0.0, 1.0)), float(np.clip(we, 0.0, 1.0))

    @staticmethod
    def _normalize_elite_mode(mode: str) -> str:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"blend", "bonus", "off"}:
            return "bonus"
        return mode_norm

    @staticmethod
    def _normalize_stability_rule(rule: str) -> str:
        rule_norm = str(rule).strip().lower()
        if rule_norm not in {"or", "and"}:
            return "or"
        return rule_norm

    @staticmethod
    def _normalize_null_mode(mode: str) -> str:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"soft", "hard", "off"}:
            return "soft"
        return mode_norm

    @staticmethod
    def _normalize_null_apply_to(target: str) -> str:
        t = str(target).strip().lower()
        if t not in {"global_score", "final_score"}:
            return "global_score"
        return t

    def _resolve_null_policy(
        self,
        *,
        low_data: bool,
    ) -> tuple[bool, str, float, int, float, str]:
        enabled = bool(getattr(self.config, "null_enabled", False))
        mode = self._normalize_null_mode(getattr(self.config, "null_mode", "soft"))
        if mode == "off":
            enabled = False
        q = float(np.clip(float(getattr(self.config, "null_quantile", 0.90)), 0.50, 0.999))
        if bool(low_data):
            n_shuffle = int(max(int(getattr(self.config, "null_shuffle_runs_low_data", 15)), 1))
            alpha = float(np.clip(float(getattr(self.config, "null_alpha_low_data", 0.08)), 0.0, 1.0))
        else:
            n_shuffle = int(max(int(getattr(self.config, "null_shuffle_runs_normal", 30)), 1))
            alpha = float(np.clip(float(getattr(self.config, "null_alpha_normal", 0.12)), 0.0, 1.0))
        apply_to = self._normalize_null_apply_to(getattr(self.config, "null_apply_to", "global_score"))
        return bool(enabled), str(mode), float(q), int(n_shuffle), float(alpha), str(apply_to)

    def _compute_null_q_series(
        self,
        *,
        features: list[str],
        perm_processed_global: pd.DataFrame,
        drop_processed_global: pd.DataFrame,
        n_shuffle: int,
        q: float,
    ) -> dict[str, pd.Series]:
        """채널별 독립 null quantile과 결합 null quantile을 함께 반환한다.

        Returns
        -------
        dict with keys: "combined", "perm", "drop"
            각각 feature-indexed pd.Series.
        """
        p = len(features)
        _zero = pd.Series(np.zeros(p), index=features, dtype=float)
        if p == 0:
            empty = pd.Series(dtype=float)
            return {"combined": empty, "perm": empty, "drop": empty}

        def _vote_matrix(df: pd.DataFrame) -> np.ndarray:
            if df is None or df.empty or ("fold" not in df.columns) or ("feature" not in df.columns):
                return np.empty((0, p), dtype=float)
            work = df.copy()
            work["vote"] = pd.to_numeric(work.get("vote", 0.0), errors="coerce").fillna(0.0)
            pv = (
                work.pivot_table(index="fold", columns="feature", values="vote", aggfunc="mean")
                .reindex(columns=features)
                .fillna(0.0)
            )
            if pv.empty:
                return np.empty((0, p), dtype=float)
            return pv.to_numpy(dtype=float)

        perm_mat = _vote_matrix(perm_processed_global)
        drop_mat = _vote_matrix(drop_processed_global)

        w_perm = float(self.config.weight_perm)
        w_drop = float(self.config.weight_drop) if bool(self.config.use_score_drop) else 0.0
        denom = w_perm + w_drop
        if denom <= 0.0:
            w_perm_n, w_drop_n = 1.0, 0.0
        else:
            w_perm_n, w_drop_n = w_perm / denom, w_drop / denom

        null_samples = np.zeros((int(n_shuffle), p), dtype=float)
        perm_null_samples = np.zeros((int(n_shuffle), p), dtype=float)
        drop_null_samples = np.zeros((int(n_shuffle), p), dtype=float)
        rng = np.random.default_rng(20260324)

        n_perm_folds = perm_mat.shape[0]
        n_drop_folds = drop_mat.shape[0]
        total_perms_per_iter = n_perm_folds + n_drop_folds
        if total_perms_per_iter == 0:
            return {"combined": _zero, "perm": _zero, "drop": _zero}

        # 모든 shuffle iteration에 필요한 permutation index를 일괄 생성
        # RNG 소비 순서 보존: 각 iteration b에서 perm_folds 먼저, drop_folds 이후
        all_orders = np.empty((int(n_shuffle) * total_perms_per_iter, p), dtype=np.intp)
        for k in range(int(n_shuffle) * total_perms_per_iter):
            all_orders[k] = rng.permutation(p)

        for b in range(int(n_shuffle)):
            base = b * total_perms_per_iter
            perm_null = np.zeros((p,), dtype=float)
            drop_null = np.zeros((p,), dtype=float)
            if n_perm_folds > 0:
                orders_b = all_orders[base:base + n_perm_folds]
                shuffled = perm_mat[np.arange(n_perm_folds)[:, None], orders_b]
                perm_null = shuffled.mean(axis=0)
            if n_drop_folds > 0:
                orders_b = all_orders[base + n_perm_folds:base + total_perms_per_iter]
                shuffled = drop_mat[np.arange(n_drop_folds)[:, None], orders_b]
                drop_null = shuffled.mean(axis=0)
            perm_null_samples[b, :] = perm_null
            drop_null_samples[b, :] = drop_null
            null_samples[b, :] = (float(w_perm_n) * perm_null) + (float(w_drop_n) * drop_null)

        q_vals = np.quantile(null_samples, float(q), axis=0)
        q_perm = np.quantile(perm_null_samples, float(q), axis=0) if n_perm_folds > 0 else np.zeros(p)
        q_drop = np.quantile(drop_null_samples, float(q), axis=0) if n_drop_folds > 0 else np.zeros(p)
        return {
            "combined": pd.Series(q_vals, index=features, dtype=float),
            "perm": pd.Series(q_perm, index=features, dtype=float),
            "drop": pd.Series(q_drop, index=features, dtype=float),
        }

    def _finalize_selection(
        self,
        *,
        global_df: pd.DataFrame,
        elite_df: pd.DataFrame,
        perm_processed_global: pd.DataFrame,
        drop_processed_global: pd.DataFrame,
        n_elite: int,
        low_data: bool,
        n_samples: int = 0,
    ) -> pd.DataFrame:
        if global_df is None or global_df.empty:
            return pd.DataFrame(columns=["feature", "selected"])

        out = global_df.copy()
        out = out.rename(
            columns={
                "scale_score": "global_score",
                "perm_vote_score": "perm_vote_score_global",
                "drop_vote_score": "drop_vote_score_global",
            }
        )

        if elite_df is not None and not elite_df.empty:
            elite = elite_df[
                [
                    "feature",
                    "scale_score",
                    "perm_vote_score",
                    "drop_vote_score",
                ]
            ].copy().rename(
                columns={
                    "scale_score": "elite_score",
                    "perm_vote_score": "perm_vote_score_elite",
                    "drop_vote_score": "drop_vote_score_elite",
                }
            )
            out = out.merge(elite, on="feature", how="left")
            out["elite_score"] = pd.to_numeric(out["elite_score"], errors="coerce").fillna(out["global_score"])
            out["perm_vote_score_elite"] = pd.to_numeric(out["perm_vote_score_elite"], errors="coerce").fillna(
                out["perm_vote_score_global"]
            )
            out["drop_vote_score_elite"] = pd.to_numeric(out["drop_vote_score_elite"], errors="coerce").fillna(
                out["drop_vote_score_global"]
            )
        else:
            out["elite_score"] = pd.to_numeric(out["global_score"], errors="coerce").fillna(0.0)
            out["perm_vote_score_elite"] = pd.to_numeric(out["perm_vote_score_global"], errors="coerce").fillna(0.0)
            out["drop_vote_score_elite"] = pd.to_numeric(out["drop_vote_score_global"], errors="coerce").fillna(0.0)

        global_score = pd.to_numeric(out["global_score"], errors="coerce").fillna(0.0)
        elite_score = pd.to_numeric(out["elite_score"], errors="coerce").fillna(0.0)
        elite_mode = self._normalize_elite_mode(self.config.elite_mode)
        out["elite_mode"] = str(elite_mode)

        if elite_mode == "blend":
            wg, we = self._resolve_scale_weights(low_data=bool(low_data), n_elite=int(n_elite))
            out["weight_global"] = float(wg)
            out["weight_elite"] = float(we)
            out["elite_bonus_beta"] = 0.0
            out["elite_bonus"] = 0.0
            out["final_score"] = (float(wg) * global_score) + (float(we) * elite_score)
        elif elite_mode == "bonus":
            beta = float(np.clip(float(self.config.elite_bonus_beta), 0.0, 1.0))
            bonus_gap = (elite_score - global_score).clip(lower=0.0)
            out["weight_global"] = 1.0
            out["weight_elite"] = 0.0
            out["elite_bonus_beta"] = float(beta)
            out["elite_bonus"] = float(beta) * bonus_gap
            out["final_score"] = global_score + out["elite_bonus"]
        else:
            out["weight_global"] = 1.0
            out["weight_elite"] = 0.0
            out["elite_bonus_beta"] = 0.0
            out["elite_bonus"] = 0.0
            out["final_score"] = global_score

        # Backward-compatible columns expected by downstream and debug plots.
        out["perm_selection_rate"] = pd.to_numeric(out["perm_vote_score_global"], errors="coerce").fillna(0.0)
        out["drop_selection_rate"] = pd.to_numeric(out["drop_vote_score_global"], errors="coerce").fillna(0.0)
        out["forced_by_constraint"] = False

        # 채널 불일치 패널티: perm/drop 불일치가 클수록 final_score 감산
        dp_enabled = bool(getattr(self.config, "disagreement_penalty_enabled", False))
        if dp_enabled:
            dp_thr = float(getattr(self.config, "disagreement_threshold", 0.25))
            dp_scale = float(getattr(self.config, "disagreement_penalty_scale", 0.5))
            disagreement = (out["perm_selection_rate"] - out["drop_selection_rate"]).abs()
            dp_penalty = (disagreement - dp_thr).clip(lower=0.0) * dp_scale
            out["disagreement"] = disagreement
            out["disagreement_penalty"] = dp_penalty
            out["final_score"] = (pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0) - dp_penalty).clip(lower=0.0)
        else:
            out["disagreement"] = (out["perm_selection_rate"] - out["drop_selection_rate"]).abs()
            out["disagreement_penalty"] = 0.0

        null_enabled, null_mode, null_q, null_runs, null_alpha, null_apply_to = self._resolve_null_policy(
            low_data=bool(low_data),
        )
        out["null_enabled"] = bool(null_enabled)
        out["null_mode"] = str(null_mode)
        out["null_quantile"] = float(null_q)
        out["null_shuffle_runs"] = int(null_runs)
        out["null_alpha"] = float(null_alpha)
        out["null_apply_to"] = str(null_apply_to)

        if bool(null_enabled):
            feature_order = out["feature"].astype(str).tolist()
            null_q_result = self._compute_null_q_series(
                features=feature_order,
                perm_processed_global=perm_processed_global,
                drop_processed_global=drop_processed_global,
                n_shuffle=int(null_runs),
                q=float(null_q),
            )
            null_q_combined = null_q_result["combined"]
            null_q_perm = null_q_result["perm"]
            null_q_drop = null_q_result["drop"]

            out["null_q"] = out["feature"].astype(str).map(null_q_combined).astype(float)
            out["null_q_perm"] = out["feature"].astype(str).map(null_q_perm).astype(float)
            out["null_q_drop"] = out["feature"].astype(str).map(null_q_drop).astype(float)

            # --- 채널별 독립 null penalty ---
            perm_vote = pd.to_numeric(out["perm_vote_score_global"], errors="coerce").fillna(0.0)
            drop_vote = pd.to_numeric(out["drop_vote_score_global"], errors="coerce").fillna(0.0)

            perm_null_margin = perm_vote - out["null_q_perm"]
            drop_null_margin = drop_vote - out["null_q_drop"]
            out["null_margin_perm"] = perm_null_margin
            out["null_margin_drop"] = drop_null_margin

            # 채널 가중치 (global_score 결합 시 사용한 것과 동일)
            w_perm = float(getattr(self.config, "weight_perm", 0.80))
            w_drop = float(getattr(self.config, "weight_drop", 0.20)) if bool(
                getattr(self.config, "use_score_drop", True)
            ) else 0.0
            w_denom = w_perm + w_drop
            if w_denom <= 0.0:
                w_perm_n, w_drop_n = 1.0, 0.0
            else:
                w_perm_n, w_drop_n = w_perm / w_denom, w_drop / w_denom

            if str(null_mode) == "hard":
                perm_ch_penalty = np.where(perm_null_margin >= 0.0, 0.0, 1e6)
                drop_ch_penalty = np.where(drop_null_margin >= 0.0, 0.0, 1e6)
            else:
                perm_ch_penalty = float(null_alpha) * np.clip(-perm_null_margin, 0.0, None)
                drop_ch_penalty = float(null_alpha) * np.clip(-drop_null_margin, 0.0, None)

            channel_penalty = (float(w_perm_n) * perm_ch_penalty) + (float(w_drop_n) * drop_ch_penalty)

            # 기존 결합 null과의 호환: 결합 margin/pass도 기록
            if str(null_apply_to) == "final_score":
                null_target_score = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)
            else:
                null_target_score = global_score
            out["null_margin"] = null_target_score - out["null_q"]
            out["null_pass"] = (perm_null_margin >= 0.0) | (drop_null_margin >= 0.0)

            out["null_penalty"] = channel_penalty
            out["final_score_adj"] = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0) - pd.to_numeric(
                out["null_penalty"], errors="coerce"
            ).fillna(0.0)
        else:
            out["null_q"] = np.nan
            out["null_q_perm"] = np.nan
            out["null_q_drop"] = np.nan
            out["null_margin"] = np.nan
            out["null_margin_perm"] = np.nan
            out["null_margin_drop"] = np.nan
            out["null_pass"] = True
            out["null_penalty"] = 0.0
            out["final_score_adj"] = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)

        tau = float(self.config.final_score_threshold)
        g_floor = float(self.config.global_score_floor)
        base_selected = (
            (pd.to_numeric(out["final_score_adj"], errors="coerce").fillna(0.0) >= tau)
            & (global_score >= g_floor)
        )

        stability_enabled = bool(getattr(self.config, "stability_enabled", True))
        very_low_n_thr = int(getattr(self.config, "stability_very_low_data_n_threshold", 50))
        if bool(low_data) and int(n_samples) < very_low_n_thr:
            stability_rule = self._normalize_stability_rule(getattr(self.config, "stability_rule_very_low_data", "or"))
            perm_thr = float(getattr(self.config, "stability_perm_min_rate_very_low_data", 0.55))
            drop_thr = float(getattr(self.config, "stability_drop_min_rate_very_low_data", 0.35))
        elif bool(low_data):
            stability_rule = self._normalize_stability_rule(getattr(self.config, "stability_rule_low_data", "and"))
            perm_thr = float(getattr(self.config, "stability_perm_min_rate_low_data", 0.60))
            drop_thr = float(getattr(self.config, "stability_drop_min_rate_low_data", 0.44))
        else:
            stability_rule = self._normalize_stability_rule(getattr(self.config, "stability_rule_normal", "and"))
            perm_thr = float(getattr(self.config, "stability_perm_min_rate_normal", 0.70))
            drop_thr = float(getattr(self.config, "stability_drop_min_rate_normal", 0.60))

        perm_thr = float(np.clip(perm_thr, 0.0, 1.0))
        drop_thr = float(np.clip(drop_thr, 0.0, 1.0))
        perm_pass = out["perm_selection_rate"] >= perm_thr
        drop_pass = out["drop_selection_rate"] >= drop_thr

        if stability_rule == "and":
            stability_pass = perm_pass & drop_pass
        else:
            stability_pass = perm_pass | drop_pass

        out["stability_enabled"] = bool(stability_enabled)
        out["stability_rule"] = str(stability_rule)
        out["stability_perm_threshold"] = float(perm_thr)
        out["stability_drop_threshold"] = float(drop_thr)
        out["stability_perm_pass"] = perm_pass.astype(bool)
        out["stability_drop_pass"] = drop_pass.astype(bool)
        out["stability_pass"] = stability_pass.astype(bool)

        if bool(stability_enabled):
            out["selected"] = base_selected & out["stability_pass"]
            out["reason"] = np.where(
                out["selected"],
                "weighted_vote_pass",
                np.where(base_selected, "stability_gate_fail", "weighted_vote_fail"),
            )
        else:
            out["selected"] = base_selected
            out["reason"] = np.where(out["selected"], "weighted_vote_pass", "weighted_vote_fail")

        out = out.sort_values(
            by=["selected", "final_score_adj", "final_score", "global_score", "delta_mean_norm"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        return out
