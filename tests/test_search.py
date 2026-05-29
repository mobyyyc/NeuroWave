import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import soundfile as sf

from minisynth.constants import DEFAULT_SAMPLE_RATE
from minisynth.io import load_patch
from minisynth.schema import SynthConfig, VECTOR_PARAMETERS
from minisynth.search import (
    format_search_progress,
    print_search_progress,
    random_search,
    random_vector,
    render_config_audio,
    save_search_result,
    search_report,
)


class TestRandomSearch(unittest.TestCase):
    def test_random_vector_is_reproducible_from_seeded_rng(self):
        first = random_vector(np.random.default_rng(123))
        second = random_vector(np.random.default_rng(123))

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(VECTOR_PARAMETERS))
        for value in first:
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 1.0)

    def test_render_config_audio_returns_audio_array(self):
        audio = render_config_audio(SynthConfig(length=1.0))

        self.assertEqual(audio.ndim, 1)
        self.assertGreater(len(audio), 0)
        self.assertTrue(np.all(np.isfinite(audio)))

    def test_random_search_returns_lowest_scoring_candidate(self):
        scores = iter([3.0, 1.0, 2.0])

        def fake_renderer(config):
            return np.array(config.to_vector())

        def fake_comparer(target_audio, target_sample_rate, candidate_audio, candidate_sample_rate):
            score = next(scores)
            return {"weighted_distance": score}

        result = random_search(
            np.zeros(10),
            DEFAULT_SAMPLE_RATE,
            iterations=3,
            seed=1,
            renderer=fake_renderer,
            comparer=fake_comparer,
        )

        self.assertEqual(result["iteration"], 1)
        self.assertEqual(result["score"], 1.0)
        self.assertIsInstance(result["config"], SynthConfig)
        self.assertEqual(len(result["vector"]), len(VECTOR_PARAMETERS))
        self.assertEqual(result["evaluations"], 3)

    def test_random_search_skips_invalid_candidates(self):
        calls = {"count": 0}

        def fake_renderer(config):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("invalid candidate")
            return np.array(config.to_vector())

        def fake_comparer(target_audio, target_sample_rate, candidate_audio, candidate_sample_rate):
            return {"weighted_distance": 1.0}

        result = random_search(
            np.zeros(10),
            iterations=1,
            seed=1,
            renderer=fake_renderer,
            comparer=fake_comparer,
        )

        self.assertEqual(result["evaluations"], 1)
        self.assertEqual(result["attempts"], 2)

    def test_random_search_reports_progress(self):
        progress = []

        def fake_renderer(config):
            return np.array(config.to_vector())

        def fake_comparer(target_audio, target_sample_rate, candidate_audio, candidate_sample_rate):
            return {"weighted_distance": 1.0}

        random_search(
            np.zeros(10),
            iterations=2,
            seed=1,
            renderer=fake_renderer,
            comparer=fake_comparer,
            progress_callback=progress.append,
        )

        self.assertEqual(len(progress), 2)
        self.assertEqual(progress[0]["evaluations"], 1)
        self.assertEqual(progress[0]["iterations"], 2)
        self.assertEqual(progress[0]["best_score"], 1.0)

    def test_random_search_respects_progress_interval(self):
        progress = []

        def fake_renderer(config):
            return np.array(config.to_vector())

        def fake_comparer(target_audio, target_sample_rate, candidate_audio, candidate_sample_rate):
            return {"weighted_distance": 1.0}

        random_search(
            np.zeros(10),
            iterations=3,
            seed=1,
            renderer=fake_renderer,
            comparer=fake_comparer,
            progress_callback=progress.append,
            progress_interval=2,
        )

        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["evaluations"], 2)

    def test_random_search_rejects_invalid_iterations(self):
        with self.assertRaises(ValueError):
            random_search(np.zeros(10), iterations=0)

    def test_random_search_rejects_invalid_progress_interval(self):
        with self.assertRaises(ValueError):
            random_search(np.zeros(10), progress_interval=0)

    def test_format_search_progress_includes_scores(self):
        text = format_search_progress(
            {
                "evaluations": 2,
                "iterations": 10,
                "attempts": 3,
                "score": 4.0,
                "best_score": 1.5,
                "improved": False,
            }
        )

        self.assertIn("evaluation 2/10", text)
        self.assertIn("score 4.000000", text)
        self.assertIn("best 1.500000", text)

    def test_print_search_progress_prints_formatted_progress(self):
        output = io.StringIO()

        with redirect_stdout(output):
            print_search_progress(
                {
                    "evaluations": 1,
                    "iterations": 1,
                    "attempts": 1,
                    "score": 0.0,
                    "best_score": 0.0,
                    "improved": True,
                }
            )

        self.assertIn("evaluation 1/1", output.getvalue())

    def test_save_search_result_writes_best_patch_and_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SynthConfig(length=1.0, osc1_wave="sine")
            audio = render_config_audio(config)
            result = {
                "score": 1.25,
                "iteration": 0,
                "attempt": 0,
                "evaluations": 1,
                "attempts": 1,
                "distances": {"weighted_distance": 1.25},
                "config": config,
                "vector": config.to_vector(),
                "audio": audio,
            }

            paths = save_search_result(result, Path(tmpdir) / "match_001")

            self.assertEqual(paths["patch_path"], Path(tmpdir) / "match_001" / "best_patch.json")
            self.assertEqual(paths["audio_path"], Path(tmpdir) / "match_001" / "best.wav")
            self.assertEqual(paths["report_path"], Path(tmpdir) / "match_001" / "report.json")
            self.assertTrue(paths["patch_path"].exists())
            self.assertTrue(paths["audio_path"].exists())
            self.assertTrue(paths["report_path"].exists())
            self.assertEqual(load_patch(paths["patch_path"])["osc1_wave"], "sine")

            info = sf.info(paths["audio_path"])
            self.assertEqual(info.samplerate, DEFAULT_SAMPLE_RATE)
            self.assertEqual(info.channels, 1)

            report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
            self.assertEqual(report["score"], 1.25)
            self.assertEqual(report["config"]["osc1_wave"], "sine")
            self.assertEqual(report["distances"]["weighted_distance"], 1.25)

    def test_search_report_serializes_run_summary(self):
        config = SynthConfig(length=1.0)
        audio = render_config_audio(config)
        result = {
            "score": 2.5,
            "iteration": 1,
            "attempt": 2,
            "evaluations": 3,
            "attempts": 4,
            "distances": {"weighted_distance": 2.5},
            "config": config,
            "vector": config.to_vector(),
            "audio": audio,
        }

        report = search_report(
            result,
            Path("runs/match_001/best_patch.json"),
            Path("runs/match_001/best.wav"),
        )

        self.assertEqual(report["score"], 2.5)
        self.assertEqual(report["iteration"], 1)
        self.assertEqual(report["evaluations"], 3)
        self.assertEqual(report["patch_path"], "runs/match_001/best_patch.json")
        self.assertEqual(report["audio_path"], "runs/match_001/best.wav")
        self.assertEqual(report["sample_rate"], DEFAULT_SAMPLE_RATE)
        self.assertEqual(report["frames"], len(audio))
        self.assertEqual(len(report["vector"]), len(VECTOR_PARAMETERS))

    def test_random_search_matches_neurowave_generated_target(self):
        target_config = SynthConfig(length=1.0, osc1_wave="saw", osc2_wave="triangle")
        target_audio = render_config_audio(target_config)

        result = random_search(
            target_audio,
            DEFAULT_SAMPLE_RATE,
            iterations=2,
            seed=5,
        )

        self.assertEqual(result["evaluations"], 2)
        self.assertTrue(np.isfinite(result["score"]))
        self.assertIsInstance(result["config"], SynthConfig)
        self.assertEqual(len(result["vector"]), len(VECTOR_PARAMETERS))
        self.assertEqual(result["audio"].ndim, 1)
        self.assertGreater(len(result["audio"]), 0)


if __name__ == "__main__":
    unittest.main()
