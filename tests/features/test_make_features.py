import numpy as np
import pytest

from src.data.make_dataset import read_data
from src.entities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer


@pytest.fixture
def feature_params(
    categorical_features: list[str],
    features_to_drop: list[str],
    numerical_features: list[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )
    return params


def test_make_features(
    feature_params: FeatureParams, dataset_path: str,
):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert features.shape[0] == data.shape[0]


def test_extract_target(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)

    target = extract_target(data, feature_params)
    assert np.array_equal(target, data[feature_params.target_col].to_numpy())
