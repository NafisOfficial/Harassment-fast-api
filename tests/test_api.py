import unittest

from fastapi.testclient import TestClient

from app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_endpoint(self):
        response = self.client.post(
            "/predict",
            json={"comment": "You are a terrible person"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("label", payload)
        self.assertIn("confidence", payload)
        self.assertIn("comment", payload)
        self.assertEqual(payload["comment"], "You are a terrible person")


if __name__ == "__main__":
    unittest.main()
