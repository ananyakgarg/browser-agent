"""Deterministic sample scoring used as the source-of-truth quality metric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from validator import evaluate_output_validity

JUDGMENT_HINTS = (
    "flag",
    "rating",
    "verdict",
    "assessment",
    "decision",
    "modified_within",
    "material_change",
    "status",
)


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"n/a", "na", "unknown", "none", "null"}


def _completion_time_score(seconds: float) -> float:
    if seconds <= 120:
        return 1.0
    if seconds <= 600:
        # 120s -> 1.0, 600s -> 0.5
        return round(1.0 - ((seconds - 120) / 480) * 0.5, 4)
    if seconds <= 1200:
        # 600s -> 0.5, 1200s -> 0.0
        return round(max(0.0, 0.5 - ((seconds - 600) / 600) * 0.5), 4)
    return 0.0


def _retry_score(retries: int) -> float:
    return round(max(0.0, 1.0 - min(retries, 5) / 5.0), 4)


def _judgment_evidence_rate(csv_columns: list[str], data: dict[str, Any]) -> float:
    judgment_columns = []
    for col in csv_columns:
        lowered = col.lower()
        if "evidence" in lowered:
            continue
        if any(hint in lowered for hint in JUDGMENT_HINTS):
            judgment_columns.append(col)

    if not judgment_columns:
        return 1.0

    with_evidence = 0
    for col in judgment_columns:
        evidence_col = f"{col}_evidence"
        if evidence_col in csv_columns and _is_meaningful(data.get(evidence_col)):
            with_evidence += 1
        elif evidence_col not in csv_columns:
            # If no companion evidence column exists, don't penalize this column.
            with_evidence += 1
    return round(with_evidence / len(judgment_columns), 4)


def _screenshot_reference_validity(sample_dir: Path, data: dict[str, Any]) -> float:
    if "evidence_screenshots" not in data:
        return 1.0

    raw = data.get("evidence_screenshots")
    if isinstance(raw, list):
        names = [str(x).strip() for x in raw if str(x).strip()]
    else:
        text = str(raw or "").strip()
        if not text:
            return 0.0
        names = [x.strip() for x in text.split(",") if x.strip()]

    if not names:
        return 0.0

    found = 0
    for name in names:
        p = sample_dir / name
        if p.exists() and p.suffix.lower() == ".png" and p.stat().st_size > 0:
            found += 1
    return round(found / len(names), 4)


def score_sample(
    sample_dir: Path,
    csv_columns: list[str],
    data: dict[str, Any] | None,
    retries: int,
    completion_time_sec: float,
) -> dict[str, Any]:
    """Compute deterministic sample score (0-100)."""
    payload = data or {}
    validity = evaluate_output_validity(sample_dir, csv_columns, payload)

    schema = float(validity["schema_pass_rate"])
    evidence = float(validity["required_evidence_present"])
    screenshot_validity = float(validity["screenshot_validity"])
    judgment_evidence = _judgment_evidence_rate(csv_columns, payload)
    screenshot_ref_validity = _screenshot_reference_validity(sample_dir, payload)
    retries_component = _retry_score(retries)
    time_component = _completion_time_score(completion_time_sec)

    evidence_combined = round((evidence + judgment_evidence + screenshot_ref_validity) / 3.0, 4)

    # Weighted deterministic score
    score = int(round(
        (schema * 40)
        + (evidence_combined * 25)
        + (screenshot_validity * 20)
        + (retries_component * 10)
        + (time_component * 5)
    ))

    result = {
        "score": max(0, min(100, score)),
        "schema_pass_rate": schema,
        "required_evidence_present": evidence_combined,
        "screenshot_validity": screenshot_validity,
        "judgment_evidence_rate": judgment_evidence,
        "screenshot_reference_validity": screenshot_ref_validity,
        "retries": retries,
        "completion_time_sec": round(completion_time_sec, 2),
        "retry_score": retries_component,
        "completion_time_score": time_component,
        "passed": (
            schema == 1.0
            and screenshot_validity == 1.0
            and not validity["errors"]
        ),
        "validity": validity,
    }

    out_path = sample_dir / "deterministic_score.json"
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result
