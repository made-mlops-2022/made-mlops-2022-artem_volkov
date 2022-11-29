from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from docker.types import Mount


DATA_DIR_NAME = "/data/raw/{{ ds }}"
PROCESSED_DIR_NAME = "/data/processed/{{ ds }}"
PREDICTIONS_PATH = "/data/predictions/{{ ds }}"

MODEL_PATH = Variable.get("MODEL_PATH")
TRANSFORMER_PATH = Variable.get("TRANSFORMER_PATH")

MOUNT_OBJ = [Mount(
    source="/Users/artem/Documents/BMSTU/vk_2_sem/ml_prod/hm01/airflow_ml_dags/data",
    target="/data",
    type='bind'
    )]


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "inference",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    # features_sensor = FileSensor(
    #     task_id="data_sensor",
    #     filepath="/opt/airflow" + DATA_DIR_NAME + "/features.csv"
    # )
    #
    # model_sensor = FileSensor(
    #     task_id="model_sensor",
    #     filepath="/opt/airflow" + MODEL_PATH
    # )
    #
    # transformer_sensor = FileSensor(
    #     task_id="transformer_sensor",
    #     filepath="/opt/airflow" + TRANSFORMER_PATH
    # )
    print(MODEL_PATH)
    print(TRANSFORMER_PATH)
    predict = DockerOperator(
        image="airflow-predict",
        command=f"--source_path {DATA_DIR_NAME} --out_path {PREDICTIONS_PATH} "
                f"--transformer_path {TRANSFORMER_PATH} --model_path {MODEL_PATH}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    predict
