"""Run from the project root:  python -m unittest discover tests -v"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
import index as backend  # noqa: E402

GOOD = {
    "Attendance_Percentage": 90, "Study_Hours_Per_Day": 5, "Previous_Marks": 85,
    "Assignment_Completion_Percentage": 90, "Sleep_Hours": 7.5, "Participation_Level": "High",
}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.client = backend.app.test_client()

    def test_csv_has_500_rows_and_all_classes(self):
        df = pd.read_csv(backend.DATA_PATH)
        self.assertEqual(len(df), 500)
        self.assertEqual(list(df.columns), backend.REQUIRED_COLUMNS)
        self.assertEqual(set(df["Performance"]), {"Low", "Average", "High"})
        self.assertFalse(df.isnull().any().any())

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_metrics_from_held_out_test_set(self):
        m = self.client.get("/api/metrics").get_json()
        self.assertEqual(m["dataset_size"], 500)
        self.assertEqual(sum(m["class_distribution"].values()), 500)
        self.assertEqual(sum(map(sum, m["confusion_matrix"])), m["test_size"])
        correct = sum(m["confusion_matrix"][i][i] for i in range(3))
        self.assertAlmostEqual(m["accuracy"], correct / m["test_size"])

    def test_predict_valid(self):
        r = self.client.post("/api/predict", json=GOOD)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn(body["prediction"], ["Low", "Average", "High"])
        self.assertTrue(body["explanation"])

    def test_predict_validation_errors(self):
        self.assertEqual(self.client.post("/api/predict", data="nope").status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={**GOOD, "Attendance_Percentage": 150}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={**GOOD, "Sleep_Hours": "abc"}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={**GOOD, "Participation_Level": "Huge"}).status_code, 422)

    def test_misc_routes(self):
        self.assertEqual(self.client.get("/api/predict").status_code, 405)
        self.assertEqual(self.client.get("/api/nothing").status_code, 404)
        self.assertEqual(self.client.get("/api/students?limit=3").get_json()["rows"].__len__(), 3)
        for path in ("/", "/style.css", "/script.js"):
            self.assertEqual(self.client.get(path).status_code, 200)


if __name__ == "__main__":
    unittest.main()
