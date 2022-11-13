import os

from py._path.local import LocalPath

from src.train_pipeline import run_train_pipeline
from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_end2end(
    tmpdir: LocalPath,
    dataset_path: str,
    categorical_features: list[str],
    numerical_features: list[str],
    target_col: str,
    features_to_drop: list[str],
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.25, random_state=42),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
        train_params=TrainingParams(model_type="XGBClassifier"),
    )
    real_model_path, metrics = run_train_pipeline(params)
    assert metrics["accuracy_score"] > 0.5
    assert metrics["precision_score"] > 0.5
    assert metrics["recall_score"] > 0.5
    assert metrics["f1_score"] > 0.5
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
