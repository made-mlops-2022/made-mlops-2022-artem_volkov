import sys
import logging
import requests
import pandas as pd
import argparse
import click


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)


@click.command(name="requests")
@click.argument("PATH_TO_DATA", default="data/data_to_predict.csv")
@click.argument("host", default="localhost")
@click.argument("port", default="8000")
@click.argument("requests_len", default="10")
def run_request(
        path_to_data: str, host: str, port: str, requests_len: int
):
    logger.info("Reading data from file")

    data = pd.read_csv(path_to_data)
    data["id"] = data.index
    request_features = list(data.columns)
    url = f"http://{host}:{port}/predict/"

    logger.info(f"Number of requests: {requests_len}")
    logger.info(f"Url: {url}")

    for i in range(int(requests_len)):
        request_data = data.iloc[i].tolist()

        logger.info(f"Request data: {request_data}")

        response = requests.get(
            url,
            json={"data": [request_data], "features": request_features}
        )

        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response data: {response.json()}")


if __name__ == "__main__":
    run_request()
