"""Smoke-import every stage package."""

import pytest

import embodiednca
import embodiednca.v0
import embodiednca.v1
import embodiednca.v2
import embodiednca.v3
import embodiednca.v3_5
import embodiednca.v4
import embodiednca.v5


def test_root_package():
    assert embodiednca.__doc__


@pytest.mark.parametrize(
    "mod",
    [
        embodiednca.v0,
        embodiednca.v1,
        embodiednca.v2,
        embodiednca.v3,
        embodiednca.v3_5,
        embodiednca.v4,
        embodiednca.v5,
    ],
)
def test_stage_package(mod):
    assert mod.__doc__
