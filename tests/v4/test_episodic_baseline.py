"""Smoke import for embodiednca.v4.episodic_baseline."""

from embodiednca.v4.episodic_baseline import run_episodic_sequence


def test_import():
    assert run_episodic_sequence is not None
