from typing import Dict, Type

from Modeler.Models.base import BaseModel
from Modeler.Models.random_forest import RandomForestModel
from Modeler.Models.xgboost import XGBoostModel


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "xgb": XGBoostModel,
    "rf": RandomForestModel,
}


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def get_model_class(name: str) -> Type[BaseModel]:
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {name}")
    return MODEL_REGISTRY[key]
