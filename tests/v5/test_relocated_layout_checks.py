"""Smoke import for embodiednca.v5.relocated_layout_checks."""

from embodiednca.v5.relocated_layout_checks import RELOCATED_WELLS


def test_import():
    assert RELOCATED_WELLS is not None
