"""Raw .d folders under tests/data, and those of the peak detector's evaluation under data/peak_eval/raw. They are not
in git, so a test that needs a missing one is skipped."""

from pathlib import Path

import pytest


DATA_DIR = Path(__file__).resolve().parent / "data"
PEAK_EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "peak_eval" / "raw"


def require(path: Path) -> Path:
    """path, or skip the calling test when the folder is absent."""
    if not path.is_dir():
        pytest.skip(f"test data not found: {path.relative_to(DATA_DIR) if path.is_relative_to(DATA_DIR) else path}")
    return path
