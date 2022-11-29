from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

VAL_SIZE = 0.25
METRICS_DIR_NAME = "/data/metrics/{{ ds }}"
GENERATE_DIR_NAME = "/data/raw/{{ ds }}"
PROCESSED_DIR_NAME = "/data/processed/{{ ds }}"
TRANSFORMER_DIR_NAME = "/data/transformer_model/{{ ds }}"
MODEL_DIR_NAME = "/data/models/{{ ds }}"
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
        "train",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    preprocess_data = DockerOperator(
        image="airflow-preprocess",
        command=f"--source_path {GENERATE_DIR_NAME} --out_path "
                f"{PROCESSED_DIR_NAME} --transform_path {TRANSFORMER_DIR_NAME}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    split_data = DockerOperator(
        image="airflow-split",
        command=f"--source_path {PROCESSED_DIR_NAME} --val_size {VAL_SIZE}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    train_model = DockerOperator(
        image="airflow-train",
        command=f"--source_path {PROCESSED_DIR_NAME} --out_path {MODEL_DIR_NAME}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    val_model = DockerOperator(
        image="airflow-validation",
        command=f"--model_source_path {MODEL_DIR_NAME} --data_source_path " \
                f"{PROCESSED_DIR_NAME} --metric_path {METRICS_DIR_NAME}",
        task_id="docker-airflow-valid",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_OBJ
    )

    preprocess_data >> split_data >> train_model >> val_model
