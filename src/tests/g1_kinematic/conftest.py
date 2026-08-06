"""Shared fixtures for the G1 kinematic suite."""
import os

import pytest

from src.sim import L20Kinematics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
REAL_DIR = os.path.join(HERE, "fixtures", "real")
G0_FIXTURES = os.path.join(os.path.dirname(HERE), "g0_unit", "fixtures")

SIDES = ("right", "left")


@pytest.fixture(scope="session")
def kin_right():
    k = L20Kinematics("right")
    yield k
    k.close()


@pytest.fixture(scope="session")
def kin_left():
    k = L20Kinematics("left")
    yield k
    k.close()


@pytest.fixture
def kin(request):
    """Parametrizable per-side harness: indirect-param a side string."""
    side = getattr(request, "param", "right")
    k = L20Kinematics(side)
    yield k
    k.close()
