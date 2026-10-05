from pathlib import Path

import pytest

from myself.scripts.analysis.run_directories import create_run_directory


def test_readable_names_increment_without_overwriting(tmp_path: Path):
    first = create_run_directory(tmp_path, "wmh", "right", "suite_full", "2026-10-05")
    (first / "metrics.json").write_text("preserve me")
    second = create_run_directory(tmp_path, "wmh", "right", "suite_full", "2026-10-05")
    assert first.name == "2026-10-05_wmh_right_suite_full_01"
    assert second.name == "2026-10-05_wmh_right_suite_full_02"
    assert (first / "metrics.json").read_text() == "preserve me"
    assert not list(second.iterdir())


def test_invalid_name_cannot_escape_output_parent(tmp_path: Path):
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, "../wmh", "right", "suite_full", "2026-10-05")
    assert not list(tmp_path.iterdir())
