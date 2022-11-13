import unittest
import requests
from unittest import mock
from fastapi.testclient import TestClient


FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "id"]

DATA = [63.0, 1.0, 0.0, 145.0, 233.0, 1.0, 2.0,
        150.0, 0.0, 2.3, 2.0, 0.0, 1.0, 6.0]


class TestInferenceModelServer(unittest.TestCase):

    def setUp(self):
        host = "localhost"
        port = "8000"
        self.url = f"http://{host}:{port}"

    def test_health(self):
        response = requests.get(self.url + "/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "Model is ready")

    def test_prediction(self):
        response = requests.get(
            self.url + "/predict/",
            json={"data": [DATA], "features": FEATURES}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(len(response.json()) > 0)

    def test_root_page(self):
        response = requests.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "Heart Disease Cleveland online")



if __name__ == "__main__":
    unittest.main()
