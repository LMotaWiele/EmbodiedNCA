"""Smoke import for embodiednca.v3_5.action_selection_v3_5."""

from embodiednca.v3_5.action_selection_v3_5 import run_v35_episode


def test_import():
    assert run_v35_episode is not None
