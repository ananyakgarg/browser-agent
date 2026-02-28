"""Progress tracking via progress.json — supports resume on restart."""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SampleStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressTracker:
    """Track per-sample status in a progress.json file."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / "progress.json"
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self._data = json.load(f)
            logger.info(f"Loaded progress for {len(self._data)} samples")

    def _save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_status(self, sample_id: str) -> SampleStatus | None:
        entry = self._data.get(sample_id)
        if entry is None:
            return None
        return SampleStatus(entry["status"])

    def set_status(self, sample_id: str, status: SampleStatus, error: str | None = None):
        if sample_id not in self._data:
            self._data[sample_id] = {"status": status.value, "attempts": 0}
        self._data[sample_id]["status"] = status.value
        if status == SampleStatus.IN_PROGRESS:
            self._data[sample_id]["attempts"] = self._data[sample_id].get("attempts", 0) + 1
        if error:
            self._data[sample_id]["error"] = error
        self._save()

    def get_attempts(self, sample_id: str) -> int:
        entry = self._data.get(sample_id)
        return entry.get("attempts", 0) if entry else 0

    def get_pending_samples(self, all_sample_ids: list[str]) -> list[str]:
        """Return sample IDs that need processing (not yet completed)."""
        pending = []
        for sid in all_sample_ids:
            status = self.get_status(sid)
            if status in (None, SampleStatus.PENDING, SampleStatus.IN_PROGRESS, SampleStatus.FAILED):
                # Reset in_progress from prior crashed runs
                if status == SampleStatus.IN_PROGRESS:
                    self.set_status(sid, SampleStatus.PENDING)
                pending.append(sid)
        return pending

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self._data.values():
            s = entry["status"]
            counts[s] = counts.get(s, 0) + 1
        return counts
