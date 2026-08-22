"""Smoke import for embodiednca.v2.self_model_v2."""

from embodiednca.v2.self_model_v2 import train_self_model_v2


def test_import():
    assert train_self_model_v2 is not None
