"""Validate worker output for a single sample."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    if text.lower() in {"n/a", "na", "unknown", "none", "null"}:
        return False
    return True


def evaluate_output_validity(
    sample_dir: Path,
    csv_columns: list[str],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic validity metrics for a sample output."""
    errors: list[str] = []

    # Schema coverage
    missing = [col for col in csv_columns if col not in data]
    present = len(csv_columns) - len(missing)
    schema_pass_rate = (present / len(csv_columns)) if csv_columns else 1.0
    if missing:
        errors.append(f"Missing output columns: {missing}")

    # Evidence coverage
    evidence_cols = [c for c in csv_columns if "evidence" in c.lower()]
    evidence_present = sum(1 for c in evidence_cols if _is_meaningful(data.get(c)))
    evidence_rate = (evidence_present / len(evidence_cols)) if evidence_cols else 1.0

    # Screenshot validity
    screenshots = list(sample_dir.glob("*.png"))
    non_empty = [s for s in screenshots if s.stat().st_size > 0]
    screenshot_validity = 1.0 if non_empty else 0.0
    if not non_empty:
        errors.append("No non-empty screenshots found")

    # Metadata presence
    meta_path = sample_dir / "metadata.json"
    metadata_exists = meta_path.exists()
    if not metadata_exists:
        errors.append("metadata.json not found")

    return {
        "errors": errors,
        "missing_columns": missing,
        "schema_pass_rate": round(schema_pass_rate, 4),
        "required_evidence_columns": evidence_cols,
        "required_evidence_present": round(evidence_rate, 4),
        "screenshots_total": len(screenshots),
        "screenshots_non_empty": len(non_empty),
        "screenshot_validity": screenshot_validity,
        "metadata_exists": metadata_exists,
    }


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
    return evaluate_output_validity(sample_dir, csv_columns, data)["errors"]


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
