"""Smoke import for embodiednca.v4.chained_protocol."""

from embodiednca.v4.chained_protocol import run_chained_sequence


def test_import():
    assert run_chained_sequence is not None
