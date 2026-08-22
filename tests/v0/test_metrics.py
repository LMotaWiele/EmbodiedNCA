"""Smoke import for embodiednca.v0.metrics."""

from embodiednca.v0.metrics import self_mass


def test_import():
    assert self_mass is not None
