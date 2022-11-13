ML in prod hm2 Volkov Artem
==============================

Start server with docker:
------------
    docker build -t online_inference_model .
    docker run -p 8000:8000 -it online_inference_model
------------

Requests to server:
------------
    python3 request.py DATA_PATH host port requests_len
    # example: python3 request.py data/data_to_predict.csv localhost 8000 50
------------


Start server from DockerHub:
------------
    docker pull polonium13/online_inference_model:v2
    docker run -p 8000:8000 -it polonium13/online_inference_model:v2
------------
