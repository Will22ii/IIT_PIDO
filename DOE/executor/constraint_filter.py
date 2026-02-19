from __future__ import annotations

import ast
import math
from typing import Any, Iterable

import numpy as np


def has_constraint_defs(constraint_defs: list | None) -> bool:
    return bool(constraint_defs)


def clamp_ratio(value: float, *, floor: float) -> float:
    return min(max(float(value), float(floor)), 1.0)


def _normalize_scope(scope: str | None) -> str:
    value = str(scope or "x_only").strip().lower()
    if value not in {"x_only", "cae_dependent"}:
        raise ValueError(f"unsupported constraint scope: {scope}")
    return value


def validate_constraint_defs(constraint_defs: list | None) -> list[dict]:
    if not constraint_defs:
        return []
    if not isinstance(constraint_defs, list):
        raise TypeError("constraint_defs must be a list.")

    normalized: list[dict] = []
    ids: set[str] = set()
    allowed_types = {"<=", "<", ">=", ">", "=="}
    for idx, raw in enumerate(constraint_defs):
        if not isinstance(raw, dict):
            raise TypeError(f"constraint_defs[{idx}] must be a dict.")

        cid = str(raw.get("id", "")).strip()
        if not cid:
            raise ValueError(f"constraint_defs[{idx}] must include non-empty 'id'.")
        if cid in ids:
            raise ValueError(f"duplicate constraint id: {cid}")
        ids.add(cid)

        ctype = str(raw.get("type", "")).strip()
        if ctype not in allowed_types:
            raise ValueError(f"constraint_defs[{idx}] has unsupported type: {ctype}")

        if "limit" not in raw:
            raise ValueError(f"constraint_defs[{idx}] missing 'limit'.")
        try:
            limit = float(raw["limit"])
        except Exception as exc:
            raise ValueError(f"constraint_defs[{idx}] invalid limit: {raw.get('limit')}") from exc
        if not np.isfinite(limit):
            raise ValueError(f"constraint_defs[{idx}] limit must be finite.")

        expr = str(raw.get("expr", "")).strip()
        if not expr:
            raise ValueError(f"constraint_defs[{idx}] missing 'expr'.")
        try:
            ast.parse(expr, mode="eval")
        except Exception as exc:
            raise ValueError(f"constraint_defs[{idx}] invalid expr syntax: {expr}") from exc

        scope = _normalize_scope(raw.get("scope", "x_only"))

        item = dict(raw)
        item["id"] = cid
        item["type"] = ctype
        item["limit"] = float(limit)
        item["expr"] = expr
        item["scope"] = scope
        item["name"] = str(raw.get("name", cid))

        if ctype == "==":
            if "eps" in raw and raw.get("eps") is not None:
                try:
                    eps = float(raw.get("eps"))
                except Exception as exc:
                    raise ValueError(f"constraint_defs[{idx}] invalid eps.") from exc
                if not np.isfinite(eps) or eps < 0:
                    raise ValueError(f"constraint_defs[{idx}] eps must be finite and >= 0.")
                item["eps"] = float(eps)
            if "eps_ratio" in raw and raw.get("eps_ratio") is not None:
                eps_ratio = float(raw.get("eps_ratio"))
                if not np.isfinite(eps_ratio) or eps_ratio < 0:
                    raise ValueError(f"constraint_defs[{idx}] eps_ratio must be finite and >= 0.")
                item["eps_ratio"] = float(eps_ratio)
            if "eps_min" in raw and raw.get("eps_min") is not None:
                eps_min = float(raw.get("eps_min"))
                if not np.isfinite(eps_min) or eps_min < 0:
                    raise ValueError(f"constraint_defs[{idx}] eps_min must be finite and >= 0.")
                item["eps_min"] = float(eps_min)

        normalized.append(item)
    return normalized


def _build_eval_env(
    x: np.ndarray,
    var_names: Iterable[str],
    *,
    env_extra: dict[str, Any] | None = None,
) -> dict:
    env = {name: float(v) for name, v in zip(var_names, x)}
    env.update(
        {
            # 벤치마크 단계: 제한된 수학 함수만 허용
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "pi": math.pi,
            "e": math.e,
        }
    )
    if env_extra:
        env.update(env_extra)
    return env


def _eval_expr(expr: str, env: dict) -> float:
    value = eval(expr, {"__builtins__": {}}, env)  # noqa: S307
    return float(value)


def _iter_constraint_defs(constraint_defs: list | None):
    if not constraint_defs:
        return []
    if not isinstance(constraint_defs, list):
        raise TypeError("constraint_defs must be a list.")
    items = []
    for idx, cdef in enumerate(constraint_defs):
        if not isinstance(cdef, dict):
            raise TypeError(f"constraint_defs[{idx}] must be a dict.")
        cid = str(cdef.get("id") or "").strip()
        if not cid:
            raise ValueError(f"constraint_defs[{idx}] must include non-empty 'id'.")
        items.append((cid, cdef))
    return items


def _eval_tolerance(cdef: dict, limit: float) -> float:
    if cdef.get("eps") is not None:
        return max(float(cdef["eps"]), 0.0)
    eps_ratio = float(cdef.get("eps_ratio", 0.01))
    eps_min = float(cdef.get("eps_min", 1e-6))
    return max(abs(limit) * eps_ratio, eps_min)


def _constraint_scale(*, value: float, limit: float) -> float:
    return max(abs(limit), abs(value), 1.0)


def _evaluate_single_constraint(
    *,
    value: float,
    limit: float,
    ctype: str,
    cdef: dict,
) -> tuple[bool, float, float]:
    scale = _constraint_scale(value=value, limit=limit)
    if ctype == "<=":
        margin = limit - value
        return bool(value <= limit), float(margin), float((value - limit) / scale)
    if ctype == "<":
        margin = limit - value
        return bool(value < limit), float(margin), float((value - limit) / scale)
    if ctype == ">=":
        margin = value - limit
        return bool(value >= limit), float(margin), float((limit - value) / scale)
    if ctype == ">":
        margin = value - limit
        return bool(value > limit), float(margin), float((limit - value) / scale)
    if ctype == "==":
        eps = _eval_tolerance(cdef, limit)
        margin = eps - abs(value - limit)
        g = (abs(value - limit) - eps) / scale
        return bool(abs(value - limit) <= eps), float(margin), float(g)
    raise ValueError(f"unsupported constraint type: {ctype}")


def evaluate_constraints_point(
    *,
    x: np.ndarray,
    var_names: Iterable[str],
    constraint_defs: list | None,
    scope: str = "all",
    env_extra: dict[str, Any] | None = None,
    fail_fast_output_missing: bool = False,
) -> tuple[dict, bool, float]:
    if not constraint_defs:
        return {}, True, float("inf")

    env = _build_eval_env(x, var_names, env_extra=env_extra)
    constraints = {}
    feasible = True
    margin_values = []

    for cname, cdef in _iter_constraint_defs(constraint_defs):
        cscope = _normalize_scope(cdef.get("scope", "x_only"))
        if scope != "all" and cscope != scope:
            continue

        ctype = str(cdef.get("type", "<=")).strip()
        limit = float(cdef.get("limit"))
        expr = str(cdef.get("expr", "")).strip()

        value = float("inf")
        g = float("inf")
        margin_i = float("-inf")
        expr_error = None
        ok = False

        try:
            value = _eval_expr(expr, env)
            if not np.isfinite(value) or not np.isfinite(limit):
                raise ValueError("constraint result is non-finite")
            ok, margin_i, g = _evaluate_single_constraint(
                value=value,
                limit=limit,
                ctype=ctype,
                cdef=cdef,
            )
        except NameError as exc:
            if fail_fast_output_missing:
                raise ValueError(
                    f"failed to evaluate constraint '{cname}' "
                    f"(scope={cscope}): missing variable in expr '{expr}'"
                ) from exc
            expr_error = str(exc)
            ok = False
            value = float("inf")
            g = float("inf")
            margin_i = float("-inf")
        except Exception as exc:
            # 합의 정책: 식 평가 에러/NaN/inf는 infeasible로 처리
            expr_error = str(exc)
            ok = False
            value = float("inf")
            g = float("inf")
            margin_i = float("-inf")

        feasible = feasible and bool(ok)
        constraints[cname] = {
            "id": str(cdef.get("id", cname)),
            "name": str(cdef.get("name", cname)),
            "scope": cscope,
            "type": ctype,
            "limit": limit,
            "value": value,
            "g": g,
            "margin": margin_i,
            "ok": bool(ok),
            "expr": expr,
            "expr_error": expr_error,
        }
        if ctype == "==":
            constraints[cname]["eps"] = float(_eval_tolerance(cdef, limit))
        margin_values.append(margin_i)

    if not margin_values:
        return constraints, True, float("inf")

    margin = float(np.min(np.asarray(margin_values, dtype=float)))
    return constraints, feasible, margin


def evaluate_constraints_batch(
    *,
    X: np.ndarray,
    var_names: Iterable[str],
    constraint_defs: list | None,
    scope: str = "all",
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.empty((0,), dtype=bool), [], np.empty((0,), dtype=float)

    if not constraint_defs:
        return (
            np.ones((X.shape[0],), dtype=bool),
            [{} for _ in range(X.shape[0])],
            np.full((X.shape[0],), float("inf"), dtype=float),
        )

    mask = np.zeros((X.shape[0],), dtype=bool)
    payloads: list[dict] = []
    margins = np.empty((X.shape[0],), dtype=float)
    for i, x in enumerate(X):
        constraints, feasible, margin = evaluate_constraints_point(
            x=x,
            var_names=var_names,
            constraint_defs=constraint_defs,
            scope=scope,
        )
        mask[i] = bool(feasible)
        payloads.append(constraints)
        margins[i] = float(margin)
    return mask, payloads, margins
