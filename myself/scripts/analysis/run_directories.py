"""Readable, collision-safe names for personal experiment runs."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def create_run_directory(
    parent: Path,
    participant: str,
    hand: str,
    experiment: str,
    run_date: str | None = None,
) -> Path:
    """Allocate YYYY-MM-DD_participant_hand_experiment_NN without overwriting."""
    tags = (participant, hand, experiment)
    if any(re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", tag) is None for tag in tags):
        raise ValueError("Run tags must use lowercase letters, digits and underscores")
    day = run_date or datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    if date.fromisoformat(day).isoformat() != day:
        raise ValueError("Run date must use YYYY-MM-DD")
    parent.mkdir(parents=True, exist_ok=True)
    stem = "_".join((day, *tags))
    sequence = 1
    while True:
        run = parent / f"{stem}_{sequence:02d}"
        try:
            run.mkdir()
            return run
        except FileExistsError:
            sequence += 1
