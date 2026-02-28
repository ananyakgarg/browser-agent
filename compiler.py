"""Compile results from per-sample metadata.json files into results.csv."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compile_results(output_dir: Path, csv_columns: list[str]) -> Path:
    """
    Walk output/samples/*/metadata.json, aggregate into results.csv.
    Returns path to the results CSV.
    """
    samples_dir = output_dir / "samples"
    results_path = output_dir / "results.csv"

    rows: list[dict[str, str]] = []

    if not samples_dir.exists():
        logger.warning(f"Samples directory not found: {samples_dir}")
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id"] + csv_columns)
            writer.writeheader()
        return results_path

    for meta_path in sorted(samples_dir.glob("*/metadata.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {meta_path}: {e}")
            continue

        sample_id = meta.get("sample_id", meta_path.parent.name)
        data = meta.get("extracted_data", {})

        row = {"sample_id": sample_id}
        for col in csv_columns:
            row[col] = str(data.get(col, ""))
        rows.append(row)

    fieldnames = ["sample_id"] + csv_columns
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Compiled {len(rows)} results to {results_path}")
    return results_path
