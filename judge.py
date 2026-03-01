"""Judge: scores pioneer traces and picks the best one for playbook distillation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def score_trace(
    trace_path: Path,
    audit_path: Path,
    result_data: dict[str, Any] | None,
    csv_columns: list[str],
) -> dict[str, Any]:
    """Score a single pioneer trace.

    Returns a dict with individual metrics and an overall score (0-100).
    Higher is better.
    """
    score = 0
    details: dict[str, Any] = {}

    # Load trace
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    # 1. Completion — did it produce data? (50 points)
    completed = result_data is not None
    details["completed"] = completed
    if completed:
        score += 50

    # 2. Field coverage — what fraction of required columns are populated? (25 points)
    if completed and csv_columns:
        # Exclude internal fields like 'playbook'
        required = [c for c in csv_columns]
        filled = sum(
            1 for c in required
            if c in result_data and result_data[c] not in (None, "", "N/A", "unknown")
        )
        coverage = filled / len(required) if required else 1.0
        details["field_coverage"] = round(coverage, 2)
        details["fields_filled"] = filled
        details["fields_required"] = len(required)
        score += int(coverage * 25)
    else:
        details["field_coverage"] = 0.0

    # 3. Efficiency — fewer iterations = better (15 points)
    iterations = len(trace)
    details["iterations"] = iterations
    if iterations > 0:
        # 1 iteration = full 15 points, 30 iterations = 0 points
        efficiency = max(0.0, 1.0 - (iterations - 1) / 29.0)
        score += int(efficiency * 15)
        details["efficiency"] = round(efficiency, 2)

    # 4. No loops — clean execution without getting stuck (5 points)
    loop_nudges = sum(1 for s in trace if s.get("loop_nudge"))
    details["loop_nudges"] = loop_nudges
    if loop_nudges == 0:
        score += 5

    # 5. Has playbook — pioneer actually wrote a guide (5 points)
    has_playbook = bool(result_data and result_data.get("playbook"))
    details["has_playbook"] = has_playbook
    if has_playbook:
        score += 5

    # Token cost from audit (informational, not scored)
    details["total_tokens"] = audit.get("total_tokens", 0)
    details["elapsed_sec"] = audit.get("elapsed_sec", 0)

    details["score"] = score
    return details


def pick_winner(
    candidates: list[dict[str, Any]],
) -> int:
    """Pick the best pioneer from scored candidates.

    Each candidate should have: {"pioneer_id": int, "score": <score_dict>, "result": <data|None>}
    Returns the index of the winner.
    """
    if not candidates:
        raise ValueError("No candidates to judge")

    # Filter to only completed candidates
    completed = [(i, c) for i, c in enumerate(candidates) if c["score"]["completed"]]

    if not completed:
        # None completed — pick the one that got furthest (most iterations = most work done)
        logger.warning("No pioneers completed. Picking the one with most progress.")
        return max(range(len(candidates)), key=lambda i: candidates[i]["score"]["iterations"])

    if len(completed) == 1:
        return completed[0][0]

    # Among completed, pick highest score
    best_idx = max(completed, key=lambda x: x[1]["score"]["score"])[0]
    return best_idx


def format_judgment(candidates: list[dict[str, Any]], winner_idx: int) -> str:
    """Format a human-readable judgment summary."""
    lines = ["Pioneer Tournament Results:", "=" * 40]

    for i, c in enumerate(candidates):
        s = c["score"]
        marker = " << WINNER" if i == winner_idx else ""
        lines.append(
            f"  Pioneer {c['pioneer_id']}: score={s['score']}/100 "
            f"completed={s['completed']} fields={s.get('field_coverage', 0):.0%} "
            f"iterations={s['iterations']} loops={s['loop_nudges']}"
            f"{marker}"
        )

    winner = candidates[winner_idx]
    lines.append(f"\nWinner: Pioneer {winner['pioneer_id']} (score {winner['score']['score']}/100)")
    return "\n".join(lines)
