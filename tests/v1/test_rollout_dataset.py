"""Smoke import for embodiednca.v1.rollout_dataset."""

from embodiednca.v1.rollout_dataset import DEFAULT_CKPT


def test_import():
    assert DEFAULT_CKPT is not None
