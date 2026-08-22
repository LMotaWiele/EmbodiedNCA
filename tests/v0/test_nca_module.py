"""Smoke import for embodiednca.v0.nca_module."""

from embodiednca.v0.nca_module import SimpleNCA


def test_import():
    assert SimpleNCA is not None
