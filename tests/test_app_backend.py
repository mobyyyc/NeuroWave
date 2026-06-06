import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
import wave

from minisynth.app_inference import AppInferenceResult


def load_app_backend_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "app_backend.py"
    spec = importlib.util.spec_from_file_location("app_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app_backend = load_app_backend_module()


def write_tiny_wav(path, sample_rate=8000):
    frames = [0, 1200, 0, -1200] * 20
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frame_bytes = b"".join(
            int(frame).to_bytes(2, "little", signed=True) for frame in frames
        )
        handle.writeframes(frame_bytes)


class RunningBackend:
    def __init__(self, predict_function, open_folder_function=None):
        self.server = app_backend.create_server(
            host="127.0.0.1",
            port=0,
            predict_function=predict_function,
            open_folder_function=open_folder_function or (lambda _path: None),
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

    def test_run_folder_paths_from_result_reads_run_dir(self):
        paths = app_backend.run_folder_paths_from_result(
            {
                "run_id": "run",
                "run_dir": "runs/app/run",
                "predicted_wav": "runs/app/run/predicted.wav",
            }
        )

        self.assertEqual(paths, [Path("runs/app/run")])

    def test_health_endpoint(self):
        with RunningBackend(lambda _payload: {}) as backend:
            status, payload = get_json(f"{backend.base_url}/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "neurowave-app-backend")

    def test_runtime_endpoint_reports_runtime_shape(self):
        with RunningBackend(lambda _payload: {}) as backend:
            status, payload = get_json(f"{backend.base_url}/runtime")

        self.assertEqual(status, 200)
        self.assertIn("python", payload)
        self.assertIn("device", payload)
        self.assertIn("cuda_available", payload)

    def test_backend_smoke_flow_registers_prediction_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_wav = root / "input.wav"
            model = root / "model.pt"
            run_dir = root / "run"
            target_crop = run_dir / "target_crop.wav"
            predicted_patch = run_dir / "predicted_patch.json"
            predicted_wav = run_dir / "predicted.wav"
            target_spectrogram = run_dir / "target_spectrogram.json"
            predicted_spectrogram = run_dir / "predicted_spectrogram.json"
            summary = run_dir / "summary.json"

            write_tiny_wav(input_wav)
            model.write_bytes(b"fake checkpoint")
            run_dir.mkdir()
            target_crop.write_bytes(input_wav.read_bytes())
            predicted_patch.write_text('{"osc1_wave": "saw"}\n', encoding="utf-8")
            predicted_wav.write_bytes(input_wav.read_bytes())
            target_spectrogram.write_text("[]\n", encoding="utf-8")
            predicted_spectrogram.write_text("[]\n", encoding="utf-8")
            summary.write_text('{"status": "ok"}\n', encoding="utf-8")
            opened = []

            def fake_predict(payload):
                self.assertEqual(Path(payload["audio_path"]), input_wav)
                self.assertEqual(Path(payload["model_path"]), model)
                self.assertEqual(payload["freq_hz"], 440)
                return {
                    "run_id": "backend_smoke",
                    "run_dir": str(run_dir),
                    "target_crop_wav": str(target_crop),
                    "predicted_patch_json": str(predicted_patch),
                    "predicted_wav": str(predicted_wav),
                    "target_spectrogram": str(target_spectrogram),
                    "predicted_spectrogram": str(predicted_spectrogram),
                    "summary": str(summary),
                }

            with RunningBackend(fake_predict, open_folder_function=opened.append) as backend:
                health_status, health = get_json(f"{backend.base_url}/health")
                runtime_status, runtime = get_json(f"{backend.base_url}/runtime")
                predict_status, prediction = post_json(
                    f"{backend.base_url}/predict",
                    {
                        "audio_path": str(input_wav),
                        "model_path": str(model),
                        "freq_hz": 440,
                        "crop_start_seconds": 0.0,
                        "crop_end_seconds": 0.01,
                    },
                )
                encoded_patch = urllib.parse.quote(str(predicted_patch), safe="")
                encoded_wav = urllib.parse.quote(str(predicted_wav), safe="")
                patch_status, _patch_headers, patch_body = get_bytes(
                    f"{backend.base_url}/artifact?path={encoded_patch}"
                )
                wav_status, wav_headers, wav_body = get_bytes(
                    f"{backend.base_url}/artifact?path={encoded_wav}"
                )
                folder_status, folder_payload = post_json(
                    f"{backend.base_url}/open-folder",
                    {"path": str(run_dir)},
                )

        self.assertEqual(health_status, 200)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(runtime_status, 200)
        self.assertIn("device", runtime)
        self.assertEqual(predict_status, 200)
        self.assertEqual(prediction["run_id"], "backend_smoke")
        self.assertEqual(patch_status, 200)
        self.assertEqual(json.loads(patch_body.decode("utf-8")), {"osc1_wave": "saw"})
        self.assertEqual(wav_status, 200)
        self.assertEqual(wav_headers["Content-Type"], "audio/wav")
        self.assertTrue(wav_body.startswith(b"RIFF"))
        self.assertEqual(folder_status, 200)
        self.assertEqual(folder_payload["status"], "ok")
        self.assertEqual(opened, [str(run_dir.resolve())])

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

    def test_open_folder_endpoint_opens_registered_run_folder(self):
        opened = []
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            def fake_predict(_payload):
                return {
                    "run_id": "folder_run",
                    "run_dir": str(run_dir),
                }

            with RunningBackend(fake_predict, open_folder_function=opened.append) as backend:
                post_json(
                    f"{backend.base_url}/predict",
                    {
                        "audio_path": "input.wav",
                        "model_path": "model.pt",
                        "freq_hz": 440,
                    },
                )
                status, payload = post_json(
                    f"{backend.base_url}/open-folder",
                    {
                        "path": str(run_dir),
                    },
                )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(opened, [str(run_dir.resolve())])

    def test_open_folder_endpoint_rejects_unregistered_folder(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            with RunningBackend(lambda _payload: {}) as backend:
                with self.assertRaises(urllib.error.HTTPError) as context:
                    post_json(
                        f"{backend.base_url}/open-folder",
                        {
                            "path": str(run_dir),
                        },
                    )
                error = context.exception
                body = json.loads(error.read().decode("utf-8"))
                error.close()

        self.assertEqual(error.code, 403)
        self.assertIn("run folder is not available", body["error"]["message"])

    def test_open_folder_endpoint_returns_404_for_missing_registered_folder(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            def fake_predict(_payload):
                return {
                    "run_id": "missing_folder_run",
                    "run_dir": str(run_dir),
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
                run_dir.rmdir()
                with self.assertRaises(urllib.error.HTTPError) as context:
                    post_json(
                        f"{backend.base_url}/open-folder",
                        {
                            "path": str(run_dir),
                        },
                    )
                error = context.exception
                body = json.loads(error.read().decode("utf-8"))
                error.close()

        self.assertEqual(error.code, 404)
        self.assertIn("run folder not found", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
