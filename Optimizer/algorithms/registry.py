from __future__ import annotations

import importlib.util
import inspect
import os
from dataclasses import dataclass, asdict
from typing import Type

from Optimizer.algorithms.base import OptimizerAlgorithm
from Optimizer.algorithms.focus_bo.algorithm import FocusBOAlgorithm


OPTIMIZER_ALGORITHM_REGISTRY: dict[str, Type[OptimizerAlgorithm]] = {
    "focus_bo": FocusBOAlgorithm,
    "default": FocusBOAlgorithm,
}
_ALGORITHM_SOURCES: dict[str, str] = {
    "focus_bo": "built_in",
    "default": "alias",
}
_DISCOVERED_CUSTOM_IDS: set[str] = set()
_DISCOVERY_DONE = False
_DISCOVERY_ERRORS: dict[str, str] = {}


@dataclass(frozen=True)
class OptimizerAlgorithmInfo:
    algorithm_id: str
    source: str
    class_path: str
    uses_runtime_api: bool


def _algorithm_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _custom_algorithm_dir() -> str:
    return os.path.join(_algorithm_root(), "custom")


def _normalize_algorithm_id(name: str) -> str:
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("Optimizer algorithm name must be non-empty.")
    if not all(ch.isalnum() or ch in {"_", "-"} for ch in key):
        raise ValueError(
            "Optimizer algorithm name may contain only letters, digits, "
            "underscore, or hyphen."
        )
    return key


def _validate_algorithm_class(
    algorithm_cls: Type[OptimizerAlgorithm],
    *,
    require_runtime: bool,
) -> None:
    if not isinstance(algorithm_cls, type):
        raise TypeError("Optimizer algorithm must be a class.")
    run_method = getattr(algorithm_cls, "run", None)
    if run_method is None:
        raise TypeError("Optimizer algorithm class must define run(...).")
    try:
        instance = algorithm_cls()
    except Exception as exc:
        raise TypeError("Optimizer algorithm class must be instantiable with no arguments.") from exc
    run_bound = getattr(instance, "run", None)
    if run_bound is None:
        raise TypeError("Optimizer algorithm instance must define run(...).")
    signature = inspect.signature(run_bound)
    required = {"config", "resolved"}
    if require_runtime:
        required.add("runtime")
    missing = sorted(name for name in required if name not in signature.parameters)
    if missing:
        raise TypeError(
            "Optimizer algorithm run(...) missing required keyword parameter(s): "
            + ", ".join(missing)
        )


def _load_custom_algorithm_file(path: str) -> None:
    module_name = f"Optimizer.algorithms.custom._auto_{os.path.splitext(os.path.basename(path))[0]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import optimizer algorithm file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    algorithm_id = getattr(module, "ALGORITHM_ID", None)
    algorithm_cls = getattr(module, "Algorithm", None)
    if algorithm_id is None:
        raise RuntimeError("Custom optimizer algorithm must define ALGORITHM_ID.")
    if algorithm_cls is None:
        raise RuntimeError("Custom optimizer algorithm must define class Algorithm.")
    key = _normalize_algorithm_id(str(algorithm_id))
    if key in {"focus_bo", "default"}:
        raise RuntimeError(f"Custom optimizer algorithm cannot override built-in id: {key}")
    _validate_algorithm_class(algorithm_cls, require_runtime=True)
    register_optimizer_algorithm(key, algorithm_cls, source="custom")
    _DISCOVERED_CUSTOM_IDS.add(key)


def discover_custom_optimizer_algorithms(*, force: bool = False) -> dict[str, str]:
    """Import custom optimizer algorithms from Optimizer/algorithms/custom.

    Bad custom files are recorded and skipped so a broken upload does not break
    built-in algorithms. If a skipped algorithm is selected, the error list is
    included in the unsupported-algorithm message.
    """
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE and not bool(force):
        return dict(_DISCOVERY_ERRORS)

    _DISCOVERY_DONE = True
    _DISCOVERY_ERRORS.clear()
    for key in list(_DISCOVERED_CUSTOM_IDS):
        OPTIMIZER_ALGORITHM_REGISTRY.pop(key, None)
        _ALGORITHM_SOURCES.pop(key, None)
    _DISCOVERED_CUSTOM_IDS.clear()
    custom_dir = _custom_algorithm_dir()
    if not os.path.isdir(custom_dir):
        return {}

    for name in sorted(os.listdir(custom_dir)):
        if not name.endswith(".py") or name.startswith("_") or name == "__init__.py":
            continue
        path = os.path.join(custom_dir, name)
        try:
            _load_custom_algorithm_file(path)
        except Exception as exc:
            _DISCOVERY_ERRORS[name] = str(exc)
    return dict(_DISCOVERY_ERRORS)


def register_optimizer_algorithm(
    name: str,
    algorithm_cls: Type[OptimizerAlgorithm],
    *,
    source: str = "registered",
) -> None:
    key = _normalize_algorithm_id(name)
    _validate_algorithm_class(algorithm_cls, require_runtime=False)
    OPTIMIZER_ALGORITHM_REGISTRY[key] = algorithm_cls
    _ALGORITHM_SOURCES[key] = str(source or "registered")


def list_optimizer_algorithm_errors() -> dict[str, str]:
    discover_custom_optimizer_algorithms()
    return dict(_DISCOVERY_ERRORS)


def list_optimizer_algorithms() -> list[str]:
    discover_custom_optimizer_algorithms()
    return sorted(OPTIMIZER_ALGORITHM_REGISTRY.keys())


def describe_optimizer_algorithms() -> dict[str, object]:
    """Return backend-friendly registry information.

    This is intentionally metadata only. It does not execute an algorithm and it
    does not expose arbitrary custom module internals beyond class identity and
    whether the algorithm expects the common runtime API.
    """
    discover_custom_optimizer_algorithms()
    algorithms: list[dict[str, object]] = []
    for key in sorted(OPTIMIZER_ALGORITHM_REGISTRY.keys()):
        cls = OPTIMIZER_ALGORITHM_REGISTRY[key]
        try:
            run_sig = inspect.signature(cls().run)
            uses_runtime_api = "runtime" in run_sig.parameters
        except Exception:
            uses_runtime_api = False
        info = OptimizerAlgorithmInfo(
            algorithm_id=str(key),
            source=str(_ALGORITHM_SOURCES.get(key, "registered")),
            class_path=f"{cls.__module__}.{cls.__qualname__}",
            uses_runtime_api=bool(uses_runtime_api),
        )
        algorithms.append(asdict(info))
    return {
        "algorithms": algorithms,
        "errors": dict(_DISCOVERY_ERRORS),
    }


def get_optimizer_algorithm(name: str | None) -> OptimizerAlgorithm:
    discover_custom_optimizer_algorithms()
    key = str(name or "focus_bo").strip().lower()
    if key not in OPTIMIZER_ALGORITHM_REGISTRY:
        errors = discover_custom_optimizer_algorithms()
        error_text = f" Custom discovery errors: {errors}" if errors else ""
        raise ValueError(
            f"Unsupported optimizer algorithm: {name}. "
            f"Available algorithms: {list_optimizer_algorithms()}."
            f"{error_text}"
        )
    return OPTIMIZER_ALGORITHM_REGISTRY[key]()
