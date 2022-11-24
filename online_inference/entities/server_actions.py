import pandas as pd
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, conlist
from typing import Union, Optional
import pickle
from fastapi import HTTPException


class HeartDiseaseQuery(BaseModel):
    data: list[conlist(Union[float, int, None], min_items=14, max_items=14)]
    features: list[str]


class HeartDiseaseResponse(BaseModel):
    id: int
    condition: int


model: Optional[Pipeline] = None


def make_predict(
    data: list[HeartDiseaseQuery], features: list[str], model: Pipeline,
) -> list[HeartDiseaseResponse]:
    data = pd.DataFrame(data, columns=features)

    if len(data) == 0:
        raise HTTPException(
            status_code=400, detail="Query is empty"
        )

    ids = [x for x in data['id']]
    data = data.drop(['id'], axis=1)
    predicts = model.predict(data)
    result = [
        HeartDiseaseResponse(id=data_id, condition=int(condition)) for data_id, condition in zip(ids, predicts)
    ]
    return result


def load_model(model_path: str) -> Pipeline:
    model = pickle.load(open(model_path, 'rb'))
    return model
