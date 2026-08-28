from __future__ import annotations

import math
from types import CodeType
from typing import Any, Iterable

import numpy as np


def has_constraint_defs(constraint_defs: list | None) -> bool:
    return bool(constraint_defs)


def clamp_ratio(value: float, *, floor: float) -> float:
    return min(max(float(value), float(floor)), 1.0)


def _normalize_scope(scope: str | None) -> str:
    # canonical scope는 pre/post만 허용한다.
    value = str(scope or "pre").strip().lower()
    if value not in {"pre", "post"}:
        raise ValueError(f"unsupported constraint scope: {scope}")
    return value


# 등식에 eps가 없을 때 쓰는 정확 일치 허용치. 물리적 공차가 아니라
# 부동소수점 오차 흡수용이다. 정수형 출력(n_modes == 3 등)은 float로 정확히
# 표현되므로 이 값으로 통과하고, 연속값은 통과하지 못한다.
EXACT_EQUALITY_EPS = 1e-9

# 등식 분류 키. validate_constraint_defs가 채운다.
EQUALITY_KIND_KEY = "equality_kind"
EQUALITY_KIND_BAND = "type1_band"      # eps 있음. 공차 있는 목표값. rejection으로 처리
EQUALITY_KIND_EXACT = "type2_exact"    # eps 없음. 구조적 등식. pre면 투영 대상


def is_type2_equality(cdef: dict) -> bool:
    """구조적 등식(eps 미지정)인가."""
    if str(cdef.get("type", "")).strip() != "==":
        return False
    return cdef.get(EQUALITY_KIND_KEY) == EQUALITY_KIND_EXACT


def pre_type2_equality_defs(constraint_defs: list | None) -> list[dict]:
    """투영으로 처리할 제약. scope=pre 이면서 Type 2 등식인 것."""
    return [
        c
        for c in (constraint_defs or [])
        if isinstance(c, dict)
        and _normalize_scope(c.get("scope", "pre")) == "pre"
        and is_type2_equality(c)
    ]


def rejection_constraint_defs(constraint_defs: list | None) -> list[dict]:
    """rejection으로 처리할 제약. 투영 대상(pre Type 2 등식)만 제외한다.

    margin과 통과율은 이 목록으로 계산해야 한다. 투영된 제약은 정의상 모든 점이
    경계 위에 있어 거리 개념이 없고, 통과율도 항상 1이 되어 신호가 죽는다.
    """
    excluded = {id(c) for c in pre_type2_equality_defs(constraint_defs)}
    return [c for c in (constraint_defs or []) if id(c) not in excluded]


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
            # 문법 검증과 code object 캐싱을 겸한다. 이후 평가는 캐시를 재사용한다.
            compile_expr(expr)
        except Exception as exc:
            raise ValueError(f"constraint_defs[{idx}] invalid expr syntax: {expr}") from exc

        scope = _normalize_scope(raw.get("scope", "pre"))

        item = dict(raw)
        item["id"] = cid
        item["type"] = ctype
        item["limit"] = float(limit)
        item["expr"] = expr
        item["scope"] = scope
        item["name"] = str(raw.get("name", cid))

        if ctype == "==":
            for legacy_key in ("eps_ratio", "eps_min"):
                if raw.get(legacy_key) is not None:
                    raise ValueError(
                        f"constraint_defs[{idx}] '{legacy_key}' is no longer supported. "
                        "Specify an absolute 'eps' instead. "
                        "A ratio needs a reference scale, and 'limit' is not one "
                        "(limit=0 carries no scale information at all). "
                        "The tolerance is domain knowledge -- give it directly, "
                        f"e.g. {{'type': '==', 'limit': {limit}, 'eps': <tolerance>}}."
                    )

            if raw.get("eps") is not None:
                try:
                    eps = float(raw.get("eps"))
                except Exception as exc:
                    raise ValueError(f"constraint_defs[{idx}] invalid eps.") from exc
                if not np.isfinite(eps) or eps < 0:
                    raise ValueError(f"constraint_defs[{idx}] eps must be finite and >= 0.")
                item["eps"] = float(eps)
                item[EQUALITY_KIND_KEY] = EQUALITY_KIND_BAND
            else:
                # eps 공란 = 구조적 등식. pre면 투영, post면 정확 일치 라벨링.
                item.pop("eps", None)
                item[EQUALITY_KIND_KEY] = EQUALITY_KIND_EXACT

        normalized.append(item)
    return normalized


# 벤치마크 단계: 제한된 수학 함수만 허용
_MATH_ENV: dict[str, Any] = {
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

_EMPTY_BUILTINS: dict[str, Any] = {"__builtins__": {}}


def _build_eval_env(
    x: np.ndarray,
    var_names: Iterable[str],
    *,
    env_extra: dict[str, Any] | None = None,
) -> dict:
    env = {name: float(v) for name, v in zip(var_names, x)}
    # 수학 함수가 동명의 변수를 덮는 기존 우선순위를 유지한다.
    env.update(_MATH_ENV)
    if env_extra:
        env.update(env_extra)
    return env


# expr 문자열 -> code object 캐시.
# eval에 문자열을 넘기면 호출마다 파싱+컴파일이 반복된다. 제약 필터는 후보 점 수 x
# 제약 수만큼 호출되는 최심부 루프라 이 비용이 실제 계산보다 크다. code object는
# 같은 바이트코드를 같은 env에서 실행하므로 결과는 비트 단위로 동일하다.
_EXPR_CODE_CACHE: dict[str, CodeType] = {}


def compile_expr(expr: str) -> CodeType:
    """expr을 code object로 컴파일한다. 같은 문자열은 1회만 컴파일된다."""
    code = _EXPR_CODE_CACHE.get(expr)
    if code is None:
        code = compile(expr, "<constraint_expr>", "eval")
        _EXPR_CODE_CACHE[expr] = code
    return code


def _eval_expr(expr: str, env: dict) -> float:
    value = eval(compile_expr(expr), _EMPTY_BUILTINS, env)  # noqa: S307
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
    """등식의 허용 밴드 폭.

    eps가 있으면(Type 1) 그 값을 그대로 쓴다. 공차는 규격/공차표에서 나오는
    도메인 지식이므로 유저만 정할 수 있다.

    eps가 없으면(Type 2) 정확 일치로 본다. EXACT_EQUALITY_EPS는 물리적 공차가
    아니라 부동소수점 오차 흡수용이다.

    과거에는 eps가 없을 때 limit의 2%(eps_ratio)를 자동으로 채웠으나 제거했다.
    그 계산은 limit 값만 보고 변수 범위를 보지 않아, limit=0이면 근거 없이 1.0을
    스케일로 써서 문제에 따라 무제약이 되거나 만족 불가능이 되었다.
    또한 자동 채움이 Type 2를 Type 1인 것처럼 처리해 구조적 등식을 밴드 rejection으로
    풀게 만들었다.
    """
    if cdef.get("eps") is not None:
        return max(float(cdef["eps"]), 0.0)
    return EXACT_EQUALITY_EPS


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
        cscope = _normalize_scope(cdef.get("scope", "pre"))
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


def equality_residuals(
    *,
    x: np.ndarray,
    var_names: Iterable[str],
    equality_defs: list | None,
) -> np.ndarray:
    """등식 잔차 h(x) - L 벡터. 투영에서 쓴다.

    식 평가가 실패하면 inf를 넣는다. 호출부는 비유한 잔차를 투영 실패로 처리한다.
    """
    defs = list(equality_defs or [])
    if not defs:
        return np.empty((0,), dtype=float)

    env = _build_eval_env(np.asarray(x, dtype=float), var_names)
    out = np.empty((len(defs),), dtype=float)
    for i, cdef in enumerate(defs):
        try:
            value = _eval_expr(str(cdef.get("expr", "")).strip(), env)
            residual = float(value) - float(cdef.get("limit"))
            out[i] = residual if np.isfinite(residual) else float("inf")
        except Exception:
            out[i] = float("inf")
    return out


def equality_tolerances(equality_defs: list | None) -> np.ndarray:
    """등식별 허용치 벡터. 투영 수렴 판정에 쓴다."""
    defs = list(equality_defs or [])
    if not defs:
        return np.empty((0,), dtype=float)
    return np.asarray(
        [_eval_tolerance(c, float(c.get("limit"))) for c in defs],
        dtype=float,
    )


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
