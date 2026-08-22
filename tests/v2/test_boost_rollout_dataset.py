"""Smoke import for embodiednca.v2.boost_rollout_dataset."""

from embodiednca.v2.boost_rollout_dataset import PERT_BOOST


def test_import():
    assert PERT_BOOST is not None
