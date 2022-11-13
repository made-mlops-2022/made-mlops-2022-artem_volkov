import os
import uvicorn
from fastapi import FastAPI
import logging

from entities import (
    HeartDiseaseQuery,
    HeartDiseaseResponse,
    make_predict,
    load_model,
)

app = FastAPI()

logger = logging.getLogger(__name__)


@app.get("/")
def read_root():
    return "Heart Disease Cleveland online"


@app.on_event("startup")
def loading_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = "PATH_TO_MODEL was not specified"
        logger.error(err)
        raise RuntimeError(err)

    model = load_model(model_path)


@app.get("/health")
def read_health():
    return "Model is not ready " if model is None else "Model is ready"


@app.get("/predict/", response_model=list[HeartDiseaseResponse])
def predict(request: HeartDiseaseQuery):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
