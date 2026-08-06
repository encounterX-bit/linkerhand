"""Fixtures for the G2 safety suite + sys.path wiring for ``helpers``.

Putting this directory on sys.path lets the test modules ``from helpers import
...`` without a test package (matching the repo's existing test layout). The
shared sampling helpers live in ``helpers.py``; only fixtures live here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.safety import SafetyFilter  # noqa: E402


@pytest.fixture(scope="session")
def filt_right():
    return SafetyFilter("right")


@pytest.fixture(scope="session")
def filt_left():
    return SafetyFilter("left")


@pytest.fixture
def filt(request):
    side = getattr(request, "param", "right")
    return SafetyFilter(side)
