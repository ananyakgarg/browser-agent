"""Validate worker output for a single sample."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_sample(
    sample_dir: Path,
    csv_columns: list[str],
    data: dict[str, Any],
) -> list[str]:
    """
    Validate extracted data for a sample. Returns list of errors (empty = valid).

    Checks:
    1. All required csv_columns present in data
    2. At least one screenshot exists and is non-empty
    3. metadata.json written
    """
    errors: list[str] = []

    # Check all required columns present
    missing = [col for col in csv_columns if col not in data]
    if missing:
        errors.append(f"Missing output columns: {missing}")

    # Check at least one screenshot
    screenshots = list(sample_dir.glob("*.png"))
    non_empty = [s for s in screenshots if s.stat().st_size > 0]
    if not non_empty:
        errors.append("No non-empty screenshots found")

    # Check metadata.json
    meta_path = sample_dir / "metadata.json"
    if not meta_path.exists():
        errors.append("metadata.json not found")

    return errors


def write_metadata(
    sample_dir: Path,
    sample_id: str,
    data: dict[str, Any],
    row: dict[str, Any],
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write metadata.json for a completed sample."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "sample_id": sample_id,
        "input_row": {k: str(v) for k, v in row.items()},
        "extracted_data": data,
        "notes": notes,
        "screenshots": [p.name for p in sample_dir.glob("*.png")],
    }
    if extra:
        meta.update(extra)
    path = sample_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
