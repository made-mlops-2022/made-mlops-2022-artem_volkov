from dataclasses import dataclass
from typing import Optional


@dataclass()
class FeatureParams:
    categorical_features: list[str]
    numerical_features: list[str]
    features_to_drop: list[str]
    target_col: Optional[str]
