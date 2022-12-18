Homework4
==============================

ML in prodaction homework 4.

Install minikube on MAC on ARM:
------------
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-darwin-arm64
    sudo install minikube-darwin-arm64 /usr/local/bin/minikube
------------
Install kubeclt
------------
    brew install kubectl
------------
Start kubernetes cluster:
------------
    minicube start
------------

Start Docker


Usage kuberctl:
------------
    kubectl apply -f filename.yaml
------------

Install Lens to administrate or use
------------
    kubectl get pods
    kubectl get replicaset
    kubectl get deployment
------------