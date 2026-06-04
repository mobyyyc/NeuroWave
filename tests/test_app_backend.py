import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request

from minisynth.app_inference import AppInferenceResult


def load_app_backend_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "app_backend.py"
    spec = importlib.util.spec_from_file_location("app_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app_backend = load_app_backend_module()


class RunningBackend:
    def __init__(self, predict_function):
        self.server = app_backend.create_server(
            host="127.0.0.1",
            port=0,
            predict_function=predict_function,
            quiet=True,
            debug_errors=False,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def get_bytes(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, dict(response.headers), response.read()


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


class TestAppBackend(unittest.TestCase):
    def test_parse_json_body_rejects_invalid_payloads(self):
        with self.assertRaises(ValueError):
            app_backend.parse_json_body(b"")
        with self.assertRaises(ValueError):
            app_backend.parse_json_body(b"[1, 2, 3]")
        with self.assertRaises(ValueError):
            app_backend.parse_json_body(b"{")

    def test_predict_response_builds_inference_request(self):
        captured = {}

        def fake_run_app_inference(request):
            captured["request"] = request
            return AppInferenceResult(
                run_id="run",
                run_dir="runs/app/run",
                input_audio="input.wav",
                input_sample_rate=44100,
                crop_start_seconds=0.0,
                crop_end_seconds=1.0,
                crop_frames=44100,
                freq_context_hz=440.0,
                model="model.pt",
                target_crop_wav="runs/app/run/target_crop.wav",
                predicted_patch_json="runs/app/run/predicted_patch.json",
                predicted_wav="runs/app/run/predicted.wav",
                target_spectrogram="runs/app/run/target_spectrogram.json",
                predicted_spectrogram="runs/app/run/predicted_spectrogram.json",
                summary="runs/app/run/summary.json",
                model_metrics={},
                warnings=[],
            )

        original = app_backend.run_app_inference
        try:
            app_backend.run_app_inference = fake_run_app_inference
            response = app_backend.predict_response(
                {
                    "audio_path": "input.wav",
                    "model_path": "model.pt",
                    "freq_hz": 440,
                }
            )
        finally:
            app_backend.run_app_inference = original

        self.assertEqual(captured["request"].audio_path, "input.wav")
        self.assertEqual(captured["request"].freq_hz, 440)
        self.assertEqual(response["run_id"], "run")

    def test_artifact_paths_from_result_reads_known_output_fields(self):
        paths = app_backend.artifact_paths_from_result(
            {
                "run_id": "run",
                "target_crop_wav": "runs/app/run/target_crop.wav",
                "predicted_patch_json": "runs/app/run/predicted_patch.json",
                "ignored": "runs/app/run/other.txt",
            }
        )

        self.assertEqual(
            paths,
            [
                Path("runs/app/run/target_crop.wav"),
                Path("runs/app/run/predicted_patch.json"),
            ],
        )

    def test_health_endpoint(self):
        with RunningBackend(lambda _payload: {}) as backend:
            status, payload = get_json(f"{backend.base_url}/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "neurowave-app-backend")

    def test_predict_endpoint_returns_prediction_payload(self):
        def fake_predict(payload):
            return {
                "run_id": "test_run",
                "received": payload,
            }

        with RunningBackend(fake_predict) as backend:
            status, payload = post_json(
                f"{backend.base_url}/predict",
                {
                    "audio_path": "input.wav",
                    "model_path": "model.pt",
                    "freq_hz": 440,
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(payload["run_id"], "test_run")
        self.assertEqual(payload["received"]["audio_path"], "input.wav")

    def test_predict_endpoint_maps_validation_errors_to_400(self):
        with RunningBackend(lambda _payload: {}) as backend:
            request = urllib.request.Request(
                f"{backend.base_url}/predict",
                data=b"{",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(request, timeout=5)
            error = context.exception
            body = json.loads(error.read().decode("utf-8"))
            error.close()

        self.assertEqual(error.code, 400)
        self.assertIn("invalid JSON", body["error"]["message"])

    def test_unknown_endpoint_returns_404(self):
        with RunningBackend(lambda _payload: {}) as backend:
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(f"{backend.base_url}/missing", timeout=5)
            error = context.exception
            body = json.loads(error.read().decode("utf-8"))
            error.close()

        self.assertEqual(error.code, 404)
        self.assertIn("unknown endpoint", body["error"]["message"])

    def test_prediction_file_not_found_maps_to_404(self):
        def fake_predict(_payload):
            raise FileNotFoundError("missing model")

        with RunningBackend(fake_predict) as backend:
            with self.assertRaises(urllib.error.HTTPError) as context:
                post_json(
                    f"{backend.base_url}/predict",
                    {
                        "audio_path": "input.wav",
                        "model_path": "missing.pt",
                        "freq_hz": 440,
                    },
                )
            error = context.exception
            body = json.loads(error.read().decode("utf-8"))
            error.close()

        self.assertEqual(error.code, 404)
        self.assertIn("missing model", body["error"]["message"])

    def test_artifact_endpoint_serves_registered_prediction_file(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "predicted_patch.json"
            artifact.write_text('{"osc1_wave": "saw"}\n', encoding="utf-8")

            def fake_predict(_payload):
                return {
                    "run_id": "artifact_run",
                    "predicted_patch_json": str(artifact),
                }

            with RunningBackend(fake_predict) as backend:
                post_json(
                    f"{backend.base_url}/predict",
                    {
                        "audio_path": "input.wav",
                        "model_path": "model.pt",
                        "freq_hz": 440,
                    },
                )
                encoded_path = urllib.parse.quote(str(artifact), safe="")
                status, headers, body = get_bytes(f"{backend.base_url}/artifact?path={encoded_path}")

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(body.decode("utf-8")), {"osc1_wave": "saw"})

    def test_artifact_endpoint_rejects_unregistered_file(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "private.json"
            artifact.write_text("{}", encoding="utf-8")

            with RunningBackend(lambda _payload: {}) as backend:
                encoded_path = urllib.parse.quote(str(artifact), safe="")
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(
                        f"{backend.base_url}/artifact?path={encoded_path}",
                        timeout=5,
                    )
                error = context.exception
                body = json.loads(error.read().decode("utf-8"))
                error.close()

        self.assertEqual(error.code, 403)
        self.assertIn("artifact is not available", body["error"]["message"])

    def test_artifact_endpoint_returns_404_for_missing_registered_file(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "predicted.wav"
            artifact.write_bytes(b"placeholder")

            def fake_predict(_payload):
                return {
                    "run_id": "missing_artifact_run",
                    "predicted_wav": str(artifact),
                }

            with RunningBackend(fake_predict) as backend:
                post_json(
                    f"{backend.base_url}/predict",
                    {
                        "audio_path": "input.wav",
                        "model_path": "model.pt",
                        "freq_hz": 440,
                    },
                )
                artifact.unlink()
                encoded_path = urllib.parse.quote(str(artifact), safe="")
                with self.assertRaises(urllib.error.HTTPError) as context:
                    urllib.request.urlopen(
                        f"{backend.base_url}/artifact?path={encoded_path}",
                        timeout=5,
                    )
                error = context.exception
                body = json.loads(error.read().decode("utf-8"))
                error.close()

        self.assertEqual(error.code, 404)
        self.assertIn("artifact file not found", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
