"""Argparse smoke: every stage runner answers --help from repo root."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "run_sanity_checks.py",
    "run_v1_checks.py",
    "run_v2_checks.py",
    "run_v3_checks.py",
    "run_v3_5_checks.py",
    "run_v3_5_diagnostics.py",
    "run_v4_checks.py",
    "run_v5_checks.py",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_help(name: str) -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "usage:" in r.stdout.lower() or "Usage:" in r.stdout
