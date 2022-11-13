import os
import pickle
from typing import Tuple
import pandas as pd
import pytest
from py._path.local import LocalPath

from src.data.make_dataset import read_data
from src.entities import TrainingParams
from src.entities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer
from src.models.fit_predict_model import (
    serialize_model,
    load_model,
    serialize_model,
    predict_model,
    train_model
)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


@pytest.fixture
def features_and_target(
    dataset_path: str, categorical_features: list[str], numerical_features: list[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=[""],
        target_col="condition",
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_predict_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    pred_target = predict_model(model, features)
    assert pred_target.shape == target.shape
    assert accuracy_score(target, pred_target) > 0.1


def test_load_model(features_and_target: Tuple[pd.DataFrame, pd.Series], tmpdir: LocalPath):
    tmp_output = tmpdir.join("model.pkl")
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())

    model_path = serialize_model(model, tmp_output)
    new_model = load_model(model_path)

    pred_target = predict_model(new_model, features)

    assert pred_target.shape == target.shape
    assert accuracy_score(target, pred_target) > 0.5
    assert isinstance(new_model, type(model))


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    check_is_fitted(model)


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 20
    model = RandomForestClassifier(n_estimators=n_estimators)
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)
