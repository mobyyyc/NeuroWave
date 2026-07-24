"""Validate a versioned NeuroWave product-benchmark manifest."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.product_benchmark import load_product_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="datasets/product_benchmark_v1.json")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = load_product_benchmark(args.benchmark)
    print(
        json.dumps(
            {
                "id": benchmark["id"],
                "case_count": len(benchmark["cases"]),
                "categories": [category["id"] for category in benchmark["categories"]],
                "source_dataset": benchmark["source_dataset"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
