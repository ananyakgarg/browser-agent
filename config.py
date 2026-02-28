"""Parse and validate task specification JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuthConfig:
    cookies_path: str | None = None


@dataclass
class TaskConfig:
    max_workers: int = 3
    max_retries: int = 2
    timeout_per_sample_sec: int = 120
    max_iterations: int = 30


@dataclass
class TaskSpec:
    task_name: str
    per_sample_instructions: str
    input_csv: str
    sample_id_column: str
    csv_columns: list[str]
    config: TaskConfig = field(default_factory=TaskConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    @property
    def output_dir(self) -> Path:
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


def load_task_spec(path: str | Path) -> TaskSpec:
    """Load and validate a task spec from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task spec not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    # Required fields
    for key in ("task_name", "per_sample_instructions", "input_csv", "sample_id_column", "csv_columns"):
        if key not in raw:
            raise ValueError(f"Missing required field in task spec: {key}")

    # Parse config
    cfg_raw = raw.get("config", {})
    config = TaskConfig(
        max_workers=cfg_raw.get("max_workers", TaskConfig.max_workers),
        max_retries=cfg_raw.get("max_retries", TaskConfig.max_retries),
        timeout_per_sample_sec=cfg_raw.get("timeout_per_sample_sec", TaskConfig.timeout_per_sample_sec),
        max_iterations=cfg_raw.get("max_iterations", TaskConfig.max_iterations),
    )

    # Parse auth
    auth_raw = raw.get("auth", {})
    auth = AuthConfig(cookies_path=auth_raw.get("cookies_path"))

    # Validate input CSV exists
    csv_path = Path(raw["input_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    return TaskSpec(
        task_name=raw["task_name"],
        per_sample_instructions=raw["per_sample_instructions"],
        input_csv=raw["input_csv"],
        sample_id_column=raw["sample_id_column"],
        csv_columns=raw["csv_columns"],
        config=config,
        auth=auth,
    )
