import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from minisynth.app_inference import (
    AppInferenceRequest,
    app_output_paths,
    crop_audio,
    crop_frame_range,
    default_run_id,
    load_mono_audio,
    run_app_inference,
    sanitize_run_component,
    validate_frequency,
)
from minisynth.constants import DEFAULT_SAMPLE_RATE


class TestAppInference(unittest.TestCase):
    def test_sanitize_run_component_keeps_paths_safe(self):
        self.assertEqual(sanitize_run_component("My Pluck!.wav"), "My_Pluck_.wav")
        self.assertEqual(sanitize_run_component("..."), "audio")

    def test_default_run_id_uses_timestamp_and_audio_stem(self):
        class FixedNow:
            def strftime(self, _format):
                return "20260603_120000"

        self.assertEqual(
            default_run_id("C:/sounds/test pluck.wav", now=FixedNow()),
            "20260603_120000_test_pluck",
        )

    def test_app_output_paths_are_deterministic(self):
        paths = app_output_paths("runs/app", "test_run")

        self.assertEqual(paths["run_dir"], Path("runs/app/test_run"))
        self.assertEqual(paths["target_crop_wav"], Path("runs/app/test_run/target_crop.wav"))
        self.assertEqual(paths["predicted_patch_json"], Path("runs/app/test_run/predicted_patch.json"))
        self.assertEqual(paths["predicted_wav"], Path("runs/app/test_run/predicted.wav"))
        self.assertEqual(paths["summary"], Path("runs/app/test_run/summary.json"))

    def test_validate_frequency_rejects_invalid_values(self):
        self.assertEqual(validate_frequency(440), 440.0)
        with self.assertRaises(ValueError):
            validate_frequency(0)
        with self.assertRaises(ValueError):
            validate_frequency(float("nan"))

    def test_crop_frame_range_validates_bounds(self):
        self.assertEqual(crop_frame_range(100, 10, 2.0, 5.0), (20, 50, 2.0, 5.0))
        with self.assertRaises(ValueError):
            crop_frame_range(100, 10, -0.1, 1.0)
        with self.assertRaises(ValueError):
            crop_frame_range(100, 10, 5.0, 5.0)
        with self.assertRaises(ValueError):
            crop_frame_range(100, 10, 0.0, 11.0)

    def test_crop_audio_returns_selected_region(self):
        audio = np.arange(10, dtype=np.float32)
        cropped, start, end = crop_audio(audio, 10, 0.2, 0.7)

        np.testing.assert_array_equal(cropped, np.asarray([2, 3, 4, 5, 6], dtype=np.float32))
        self.assertEqual(start, 0.2)
        self.assertEqual(end, 0.7)

    def test_load_mono_audio_averages_stereo(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            stereo = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
            sf.write(path, stereo, DEFAULT_SAMPLE_RATE)

            audio, sample_rate = load_mono_audio(path)

        self.assertEqual(sample_rate, DEFAULT_SAMPLE_RATE)
        np.testing.assert_allclose(
            audio,
            np.asarray([0.5, 0.5], dtype=np.float32),
            atol=1e-4,
        )

    def test_load_mono_audio_rejects_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_mono_audio("missing.wav")

    def test_load_mono_audio_rejects_non_finite_samples(self):
        with patch("minisynth.app_inference.Path.exists", return_value=True):
            with patch(
                "minisynth.app_inference.sf.read",
                return_value=(np.asarray([np.nan], dtype=np.float32), DEFAULT_SAMPLE_RATE),
            ):
                with self.assertRaises(ValueError):
                    load_mono_audio("bad.wav")

    def test_run_app_inference_rejects_missing_model(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "target.wav"
            sf.write(audio_path, np.zeros(DEFAULT_SAMPLE_RATE // 10), DEFAULT_SAMPLE_RATE)

            request = AppInferenceRequest(
                audio_path=str(audio_path),
                model_path=str(root / "missing.pt"),
                freq_hz=440.0,
                output_dir=str(root / "runs"),
                run_id="missing_model",
            )

            with self.assertRaises(FileNotFoundError):
                run_app_inference(request)

    def test_run_app_inference_writes_expected_artifacts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "target.wav"
            model_path = root / "model.pt"
            model_path.write_bytes(b"placeholder")
            t = np.linspace(0.0, 0.5, DEFAULT_SAMPLE_RATE // 2, endpoint=False)
            source_audio = (0.05 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
            sf.write(audio_path, source_audio, DEFAULT_SAMPLE_RATE)

            request = AppInferenceRequest(
                audio_path=str(audio_path),
                model_path=str(model_path),
                freq_hz=440.0,
                crop_start_seconds=0.1,
                crop_end_seconds=0.3,
                output_dir=str(root / "runs"),
                run_id="manual_test",
                device="cpu",
            )
            predicted_patch = {
                "freq": 440.0,
                "length": 0.2,
                "osc1_wave": "saw",
                "osc1_level": 0.4,
                "osc2_wave": "sine",
                "osc2_level": 0.1,
                "osc2_detune": 0.0,
                "cutoff": 1200.0,
                "resonance": 0.2,
                "attack": 0.01,
                "decay": 0.1,
                "sustain": 0.5,
                "release": 0.1,
            }

            with patch("minisynth.app_inference.load_torch_checkpoint") as load_checkpoint:
                with patch("minisynth.app_inference.predict_patch_from_audio") as predict_patch:
                    load_checkpoint.return_value = {
                        "model": Mock(),
                        "metrics": {"model_id": "test_model"},
                    }
                    predict_patch.return_value = predicted_patch

                    result = run_app_inference(request)

            self.assertEqual(result.run_id, "manual_test")
            self.assertEqual(result.freq_context_hz, 440.0)
            self.assertEqual(result.model_metrics["model_id"], "test_model")
            self.assertTrue(Path(result.target_crop_wav).exists())
            self.assertTrue(Path(result.predicted_patch_json).exists())
            self.assertTrue(Path(result.predicted_wav).exists())
            self.assertTrue(Path(result.target_spectrogram).exists())
            self.assertTrue(Path(result.predicted_spectrogram).exists())
            self.assertTrue(Path(result.summary).exists())

            with Path(result.summary).open("r", encoding="utf-8") as file:
                summary = json.load(file)
            self.assertEqual(summary["run_id"], "manual_test")
            self.assertEqual(summary["crop_start_seconds"], 0.1)
            self.assertEqual(summary["crop_end_seconds"], 0.3)


if __name__ == "__main__":
    unittest.main()
