import os
import pytest


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "test_sample.csv")


@pytest.fixture()
def target_col():
    return "condition"


@pytest.fixture()
def categorical_features() -> list[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture
def numerical_features() -> list[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture()
def features_to_drop() -> list[str]:
    return [""]
