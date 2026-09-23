"""Pytest fixtures exposing the shared builders."""

from __future__ import annotations

import pytest
from helpers import LANES, FakeHub, make_hw, make_model  # noqa: F401


@pytest.fixture
def lanes():
    return LANES


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    """Keep the suite hermetic.

    A developer's real backend/.env must not change test outcomes, and tests
    must never reach the Hub with real credentials. Tests that need a token
    set one explicitly.
    """
    from indic_runner import config

    for name in config.HF_TOKEN_VARS:
        monkeypatch.delenv(name, raising=False)
