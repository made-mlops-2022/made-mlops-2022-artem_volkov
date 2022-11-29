import os
import pandas as pd
import pickle
import click

DATA_FILENAME = "features.csv"
PREDICT_FILENAME = "predict.csv"


@click.command("predict")
@click.option("--source_path", default="../data/raw")
@click.option("--out_path", default="../data/predictions")
@click.option("--transformer_path", default="../data/transformer_model/transform.pkl")
@click.option("--model_path", default="../data/models/model.pkl")
def predict(source_path: str, out_path: str, transformer_path: str, model_path: str) -> None:
    os.makedirs(source_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    data = pd.read_csv(os.path.join(source_path, DATA_FILENAME))

    with open(transformer_path, 'rb') as fin:
        transformer = pickle.load(fin)

    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    transform_data = pd.DataFrame(transformer.fit_transform(data))

    predictions = pd.DataFrame(model.predict(transform_data))

    predictions.to_csv(os.path.join(out_path, PREDICT_FILENAME), index=False)


if __name__ == '__main__':
    predict()
