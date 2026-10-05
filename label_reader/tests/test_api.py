import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app import app  # noqa: E402

DEMO = os.path.join(ROOT, "demo")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, name, mimetype="image/png"):
        with open(os.path.join(DEMO, name), "rb") as fh:
            data = {"image": (io.BytesIO(fh.read()), name, mimetype)}
        return self.client.post("/api/read-label", data=data, content_type="multipart/form-data")

    def test_motor_label(self):
        r = self.post("motor.png")
        self.assertEqual(r.status_code, 200)
        f = r.get_json()["fields"]
        self.assertEqual(f["serial_number"], "SN-2024-778213")
        self.assertEqual(f["voltage"], "415 VAC")
        self.assertEqual(f["power"], "7.5 kW")

    def test_pump_label(self):
        f = self.post("pump.png").get_json()["fields"]
        self.assertEqual(f["power"], "3 HP")
        self.assertEqual(f["frequency"], "60 Hz")

    def test_transformer_voltage(self):
        f = self.post("transformer.png").get_json()["fields"]
        self.assertEqual(f["voltage"], "11000 V")

    def test_rejects_non_image(self):
        r = self.client.post("/api/read-label",
                             data={"image": (io.BytesIO(b"hello"), "a.txt", "text/plain")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 415)

    def test_missing_file(self):
        self.assertEqual(self.client.post("/api/read-label").status_code, 400)

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json()["status"], "ok")

    def test_web_page_and_sample(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        r = self.client.post("/read", data={"sample": "motor"})
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("SN-2024-778213", html)
        self.assertIn("7 of 7 found", html)

    def test_web_upload_validation(self):
        self.assertEqual(self.client.post("/read").status_code, 400)

    def test_csv_export_uses_edited_values(self):
        r = self.client.post("/export/csv", data={"f_model": "FIXED-1", "f_power": "5 kW"})
        self.assertEqual(r.mimetype, "text/csv")
        self.assertIn("FIXED-1", r.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
