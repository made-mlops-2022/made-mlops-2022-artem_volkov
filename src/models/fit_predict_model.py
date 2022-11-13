import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from src.entities.train_params import TrainingParams

ClassificationModel = Union[RandomForestClassifier, XGBClassifier]
SklearnRegressionModel = Union[RandomForestClassifier, XGBClassifier]


def predict_model(
    model: Pipeline, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series,
) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(target, predicts),
        "precision_score": precision_score(target, predicts),
        "recall_score": recall_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def create_inference_pipeline(
    model: ClassificationModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> ClassificationModel :
    if train_params.model_type == "XGBClassifier":
        model = XGBClassifier(eval_metric="logloss", use_label_encoder=False, seed=train_params.random_state)
    elif train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=train_params.random_state)
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def load_model(
    model_path: str
) -> ClassificationModel:
    model = pickle.load(open(model_path, 'rb'))
    return model
