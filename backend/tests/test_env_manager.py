"""Torch variant selection and env idempotency."""

from __future__ import annotations

import pytest

from indic_runner.setup import env_manager as em


@pytest.mark.parametrize(
    "accelerator,cuda_version,expected",
    [
        ("cpu", None, "https://download.pytorch.org/whl/cpu"),
        ("mps", None, None),
        ("cuda", "12.4", "https://download.pytorch.org/whl/cu124"),
        ("cuda", "11.8", "https://download.pytorch.org/whl/cu118"),
        ("cuda", None, "https://download.pytorch.org/whl/cu124"),
    ],
)
def test_torch_index_follows_the_accelerator(accelerator, cuda_version, expected):
    assert em.torch_index_for(accelerator, cuda_version) == expected


def test_cpu_never_gets_the_cuda_index():
    """Plain PyPI would drag the whole nvidia stack onto a CPU-only host."""
    index = em.torch_index_for("cpu")
    assert index is not None and "cu1" not in index


def test_macos_uses_pypi_because_wheels_carry_mps():
    assert em.torch_index_for("mps") is None


def test_env_paths_are_under_the_framework_home():
    assert em.env_path("gguf-convert").parent == em.DIRS["envs"]
    assert em.python_path("gguf-convert").name in ("python", "python.exe")


def test_env_bin_resolves_console_scripts():
    path = em.env_bin("ct2-convert", "ct2-transformers-converter")
    assert path.name.startswith("ct2-transformers-converter")


def test_run_in_env_rejects_an_unprovisioned_env(tmp_path, monkeypatch):
    monkeypatch.setitem(em.DIRS, "envs", tmp_path / "envs")
    with pytest.raises(em.EnvProvisionError, match="not been provisioned"):
        em.run_in_env("never-made", ["-c", "print(1)"])


def test_fingerprint_is_stable_and_order_independent():
    a = em._fingerprint(("torch", "numpy"), "https://x")
    b = em._fingerprint(("numpy", "torch"), "https://x")
    assert a == b


def test_fingerprint_changes_with_the_index():
    a = em._fingerprint(("torch",), "https://download.pytorch.org/whl/cpu")
    b = em._fingerprint(("torch",), "https://download.pytorch.org/whl/cu124")
    assert a != b


def test_existing_env_with_matching_fingerprint_is_reused(tmp_path, monkeypatch):
    """A repeat setup must not reinstall a multi-GB toolchain."""
    import json

    monkeypatch.setitem(em.DIRS, "envs", tmp_path / "envs")
    packages = ("torch",)
    index = em.torch_index_for("cpu")

    base = em.env_path("reuse-me")
    interpreter = em.python_path("reuse-me")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\n")
    (base / ".indic-runner-env.json").write_text(
        json.dumps({"fingerprint": em._fingerprint(packages, index)})
    )

    def explode(*args, **kwargs):
        raise AssertionError("env was rebuilt despite a matching fingerprint")

    monkeypatch.setattr(em, "_run", explode)
    assert em.ensure_env("reuse-me", packages, accelerator="cpu") == interpreter


def test_changed_packages_trigger_a_rebuild(tmp_path, monkeypatch):
    import json

    monkeypatch.setitem(em.DIRS, "envs", tmp_path / "envs")
    base = em.env_path("rebuild-me")
    interpreter = em.python_path("rebuild-me")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\n")
    (base / ".indic-runner-env.json").write_text(json.dumps({"fingerprint": "stale"}))

    calls = []
    monkeypatch.setattr(em, "_run", lambda argv, label: calls.append(label))
    monkeypatch.setattr(em, "_uv", lambda: "/usr/bin/uv")
    em.ensure_env("rebuild-me", ("torch", "numpy"), accelerator="cpu")
    assert any("creating env" in c for c in calls)


def test_torch_is_installed_before_other_packages(tmp_path, monkeypatch):
    """Order matters: an extra index lets PyPI's CUDA torch win resolution."""
    monkeypatch.setitem(em.DIRS, "envs", tmp_path / "envs")
    monkeypatch.setattr(em, "_uv", lambda: "/usr/bin/uv")

    commands = []

    def record(argv, label):
        commands.append(argv)
        if "venv" in argv:  # emulate what `uv venv` creates on disk
            em.env_path("ordered").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(em, "_run", record)
    em.ensure_env("ordered", ("transformers", "torch"), accelerator="cpu")

    installs = [c for c in commands if "install" in c]
    assert "torch" in installs[0]
    assert "--index-url" in installs[0]
    assert "transformers" in installs[1]
    assert "--index-url" not in installs[1]


def test_version_pins_are_part_of_the_fingerprint():
    """Changing a pin must rebuild the env, not silently reuse a stale one."""
    unpinned = em._fingerprint(("transformers",), None)
    pinned = em._fingerprint(("transformers>=4.40,<5",), None)
    assert unpinned != pinned


def test_extra_python_path_reaches_the_subprocess(tmp_path, monkeypatch):
    """The vendored conversion tree is importable without being installed."""
    monkeypatch.setitem(em.DIRS, "envs", tmp_path / "envs")
    interpreter = em.python_path("probe")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\n")

    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs)

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    em.run_in_env("probe", ["-c", "pass"], extra_python_path=["/vendor", "/vendor/gguf-py"])
    assert "/vendor" in captured["env"]["PYTHONPATH"]
    assert "/vendor/gguf-py" in captured["env"]["PYTHONPATH"]
