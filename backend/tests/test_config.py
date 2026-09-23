"""`.env` loading and token resolution."""

from __future__ import annotations

import pytest

from indic_runner import config


def test_parses_plain_assignments():
    assert config._parse_env("HF_TOKEN=hf_abc\nOTHER=1\n") == {
        "HF_TOKEN": "hf_abc", "OTHER": "1"
    }


def test_ignores_comments_and_blank_lines():
    assert config._parse_env("# a comment\n\n  \nHF_TOKEN=x\n") == {"HF_TOKEN": "x"}


def test_strips_export_prefix():
    assert config._parse_env("export HF_TOKEN=x\n") == {"HF_TOKEN": "x"}


@pytest.mark.parametrize("quoted", ['"spaced value"', "'spaced value'"])
def test_strips_matching_quotes(quoted):
    assert config._parse_env(f"K={quoted}\n") == {"K": "spaced value"}


def test_ignores_lines_without_an_equals_sign():
    assert config._parse_env("JUST_A_WORD\n") == {}


def test_finds_env_in_a_parent_directory(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=x\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert config.find_env_file(nested) == tmp_path / ".env"


def test_returns_none_when_no_env_exists(tmp_path):
    assert config.find_env_file(tmp_path) is None


def test_load_env_sets_variables(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("INDIC_TEST_VAR=from_file\n")
    monkeypatch.delenv("INDIC_TEST_VAR", raising=False)
    applied = config.load_env(tmp_path)
    assert applied == {"INDIC_TEST_VAR": "from_file"}


def test_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    """An exported token must never be silently replaced by a stale file."""
    (tmp_path / ".env").write_text("INDIC_TEST_VAR=from_file\n")
    monkeypatch.setenv("INDIC_TEST_VAR", "from_shell")
    config.load_env(tmp_path)
    import os

    assert os.environ["INDIC_TEST_VAR"] == "from_shell"


def test_blank_value_does_not_mask_a_working_token(tmp_path, monkeypatch):
    """.env.example ships HF_TOKEN= empty; copying it must not clear a token."""
    (tmp_path / ".env").write_text("HF_TOKEN=\n")
    monkeypatch.setenv("HF_TOKEN", "hf_real")
    config.load_env(tmp_path, override=True)
    assert config.hf_token() == "hf_real"


def test_unreadable_env_is_not_fatal(tmp_path):
    env = tmp_path / ".env"
    env.mkdir()  # a directory where a file is expected
    assert config.load_env(tmp_path) == {}


@pytest.mark.parametrize("var", config.HF_TOKEN_VARS)
def test_token_is_read_from_each_supported_variable(var, monkeypatch):
    for name in config.HF_TOKEN_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(var, "hf_value")
    assert config.hf_token() == "hf_value"


def test_token_absent_returns_none(monkeypatch):
    for name in config.HF_TOKEN_VARS:
        monkeypatch.delenv(name, raising=False)
    assert config.hf_token() is None
