"""Validation helpers for versioned NeuroWave product benchmarks."""

import json
from pathlib import Path


REQUIRED_CASE_FIELDS = ("id", "source", "pitch_context_hz", "categories", "known_limitation")


def load_product_benchmark(path):
    """Load and validate a source-controlled product-benchmark manifest."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_product_benchmark(payload, source)
    return payload


def validate_product_benchmark(payload, source=None):
    """Validate manifest identity, category coverage, and deterministic references."""
    if not isinstance(payload, dict):
        raise ValueError("product benchmark manifest must be a JSON object")
    if not isinstance(payload.get("id"), str) or not payload["id"]:
        raise ValueError("product benchmark manifest requires an id")
    if not isinstance(payload.get("format_version"), int):
        raise ValueError("product benchmark manifest requires an integer format_version")

    source_dataset = payload.get("source_dataset")
    if not isinstance(source_dataset, dict):
        raise ValueError("product benchmark manifest requires source_dataset")
    if source_dataset.get("partition") != "benchmark":
        raise ValueError("product benchmark must reference an immutable benchmark partition")
    for name in ("release", "root"):
        if not isinstance(source_dataset.get(name), str) or not source_dataset[name]:
            raise ValueError(f"source_dataset requires {name}")
    if not isinstance(source_dataset.get("seed_start"), int) or not isinstance(
        source_dataset.get("seed_end"), int
    ):
        raise ValueError("source_dataset requires integer seed_start and seed_end")

    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("product benchmark manifest requires categories")
    category_ids = []
    for category in categories:
        if not isinstance(category, dict) or not isinstance(category.get("id"), str):
            raise ValueError("each product benchmark category requires an id")
        category_ids.append(category["id"])
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("product benchmark category ids must be unique")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("product benchmark manifest requires cases")
    case_ids = set()
    case_seeds = set()
    coverage = {category_id: 0 for category_id in category_ids}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("product benchmark cases must be objects")
        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                raise ValueError(f"product benchmark case requires {field}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("product benchmark case ids must be unique non-empty strings")
        case_ids.add(case_id)
        source_ref = case["source"]
        if not isinstance(source_ref, dict):
            raise ValueError("product benchmark case source must be an object")
        if source_ref.get("release") != source_dataset["release"]:
            raise ValueError("product benchmark case source release must match source_dataset")
        if source_ref.get("partition") != source_dataset["partition"]:
            raise ValueError("product benchmark case source partition must match source_dataset")
        seed = source_ref.get("seed")
        index = source_ref.get("index")
        if not isinstance(seed, int) or not isinstance(index, int):
            raise ValueError("product benchmark case source requires integer seed and index")
        if not source_dataset["seed_start"] <= seed <= source_dataset["seed_end"]:
            raise ValueError("product benchmark case seed is outside the declared source range")
        if index != seed - source_dataset["seed_start"]:
            raise ValueError("product benchmark case index must match its deterministic seed")
        if seed in case_seeds:
            raise ValueError("product benchmark cases must not reuse source seeds")
        case_seeds.add(seed)
        if not isinstance(case["pitch_context_hz"], (int, float)) or case["pitch_context_hz"] <= 0:
            raise ValueError("product benchmark case pitch_context_hz must be positive")
        labels = case["categories"]
        if not isinstance(labels, list) or not labels:
            raise ValueError("product benchmark case categories must be a non-empty list")
        for label in labels:
            if label not in coverage:
                raise ValueError(f"product benchmark case has unknown category: {label}")
            coverage[label] += 1
        if not isinstance(case["known_limitation"], str) or not case["known_limitation"].strip():
            raise ValueError("product benchmark case requires a known_limitation")

    expected_per_category = payload.get("expected_cases_per_category")
    if not isinstance(expected_per_category, int) or expected_per_category < 1:
        raise ValueError("product benchmark requires positive expected_cases_per_category")
    if any(count != expected_per_category for count in coverage.values()):
        raise ValueError("product benchmark category coverage does not match expected_cases_per_category")
    return True
