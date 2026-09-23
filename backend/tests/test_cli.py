"""CLI surface: the flags that exist, and the ones that deliberately do not."""

from __future__ import annotations

import pytest

from indic_runner.cli import build_parser


def test_setup_accepts_a_model_and_the_three_flags():
    args = build_parser().parse_args(
        ["setup", "sarvam-translate", "--task", "translation", "--dry-run", "--force"]
    )
    assert args.model == "sarvam-translate"
    assert args.task == "translation"
    assert args.dry_run is True
    assert args.force is True


def test_model_is_the_only_required_setup_argument():
    args = build_parser().parse_args(["setup", "sarvam-translate"])
    assert args.task is None
    assert args.dry_run is False


@pytest.mark.parametrize("flag", ["--device", "--precision", "--engine", "--models-dir"])
def test_hardware_and_path_flags_are_deliberately_absent(flag):
    """Tiering is deterministic and internal; the CLI takes no such knobs."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["setup", "alias", flag, "cpu"])


def test_task_choices_are_the_four_runner_tasks():
    action = next(
        a for a in build_parser()._subparsers._group_actions[0].choices["setup"]._actions
        if a.dest == "task"
    )
    assert set(action.choices) == {"reasoning", "summarization", "translation", "ocr"}


def test_an_invalid_task_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["setup", "alias", "--task", "evaluation"])


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# --- credentials -----------------------------------------------------------

def test_auth_line_reports_absence_without_inventing_a_token(monkeypatch):
    from indic_runner import cli, config

    for var in config.HF_TOKEN_VARS:
        monkeypatch.delenv(var, raising=False)
    line = cli._auth_line()
    assert "none" in line
    assert ".env.example" in line  # tells the user where to look


def test_auth_line_never_prints_the_token(monkeypatch):
    from indic_runner import cli

    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    line = cli._auth_line()
    assert "hf_supersecret_value" not in line
    assert "configured" in line
