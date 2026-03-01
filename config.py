"""Task spec dataclasses — built by the planning call, not from JSON files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskConfig:
    max_workers: int = 3
    max_retries: int = 3
    timeout_per_sample_sec: int = 600
    max_iterations: int = 30
    pioneer_enabled: bool = True
    num_pioneers: int = 1  # tournament: run N pioneers, judge picks best


@dataclass
class TaskSpec:
    task_name: str
    per_sample_instructions: str
    input_csv: str
    sample_id_column: str
    csv_columns: list[str]
    resolve_instructions: str = ""
    config: TaskConfig = field(default_factory=TaskConfig)
    storage_state_path: str | None = None
    output_dir_override: str | None = None

    @property
    def output_dir(self) -> Path:
        if self.output_dir_override:
            return Path(self.output_dir_override)
        return Path("output") / self.task_name

    @property
    def samples_dir(self) -> Path:
        return self.output_dir / "samples"

    def sample_dir(self, sample_id: str) -> Path:
        safe_id = str(sample_id).replace("/", "_").replace("\\", "_")
        return self.samples_dir / safe_id

    def render_instructions(self, row: dict[str, Any]) -> str:
        """Replace {column_name} placeholders with actual CSV values."""
        text = self.per_sample_instructions
        for key, value in row.items():
            text = text.replace(f"{{{key}}}", str(value))
        return text
