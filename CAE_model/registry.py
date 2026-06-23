# CAE model registry

from typing import Callable, Dict, Tuple

# 각 CAE 모델 import
from CAE_model.goldstein_price import (
    get_problem_spec as gp_spec,
    evaluate as gp_eval,
)
from CAE_model.goldstein_price_nodummy import (
    get_problem_spec as gp_nd_spec,
    evaluate as gp_nd_eval,
)

from CAE_model.rosenbrock import (
    get_problem_spec as rb_spec,
    evaluate as rb_eval,
)
from CAE_model.rosenbrock_nodummy import (
    get_problem_spec as rb_nd_spec,
    evaluate as rb_nd_eval,
)

from CAE_model.six_hump_camel import (
    get_problem_spec as sh_spec,
    evaluate as sh_eval,
)
from CAE_model.six_hump_camel_nodummy import (
    get_problem_spec as sh_nd_spec,
    evaluate as sh_nd_eval,
)

from CAE_model.cantilever_beam import (
    get_problem_spec as cb_spec,
    evaluate as cb_eval,
)
from CAE_model.cantilever_beam_nodummy import (
    get_problem_spec as cb_nd_spec,
    evaluate as cb_nd_eval,
)


# 타입 정의
ProblemSpecFunc = Callable[[], dict]
EvaluateFunc = Callable[..., dict]

CAEEntry = Tuple[ProblemSpecFunc, EvaluateFunc]


# -------------------------------------------------
# CAE registry
# -------------------------------------------------
CAE_REGISTRY: Dict[str, CAEEntry] = {
    "goldstein_price": (gp_spec, gp_eval),
    "goldstein_price_nodummy": (gp_nd_spec, gp_nd_eval),
    "rosenbrock": (rb_spec, rb_eval),
    "rosenbrock_nodummy": (rb_nd_spec, rb_nd_eval),
    "six_hump_camel": (sh_spec, sh_eval),
    "six_hump_camel_nodummy": (sh_nd_spec, sh_nd_eval),
    "cantilever_beam": (cb_spec, cb_eval),
    "cantilever_beam_nodummy": (cb_nd_spec, cb_nd_eval),
}


def get_cae(name: str) -> CAEEntry:
    """
    Retrieve CAE (problem_spec, evaluate) by name
    """
    try:
        return CAE_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unsupported CAE model: '{name}'. "
            f"Available models: {list(CAE_REGISTRY.keys())}"
        )
