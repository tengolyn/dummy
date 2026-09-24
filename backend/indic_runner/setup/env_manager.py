"""Creates and reuses isolated uv virtualenvs under ~/.indic-runner/envs.

Heavy, mutually incompatible toolchains (torch, CTranslate2, vLLM, OCR stacks)
each get their own env so they never share a dependency resolution with the
orchestrator or with each other.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from indic_runner.config import DIRS

# Torch publishes one wheel per accelerator. Installing the PyPI default on a
# machine without CUDA drags in the whole nvidia-* stack (multiple GB) that the
# host can never use, so the index is chosen from the detected accelerator.
TORCH_INDEX_BY_ACCELERATOR = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "mps": None,  # macOS wheels on PyPI already carry MPS support.
    "cuda": "https://download.pytorch.org/whl/cu124",
}

TORCH_PACKAGES = ("torch", "torchvision")

_MARKER = ".indic-runner-env.json"


class EnvProvisionError(RuntimeError):
    """Raised when an isolated environment could not be built."""


def _uv() -> str:
    found = shutil.which("uv")
    if found is None:
        raise EnvProvisionError(
            "uv not found on PATH; Indic-Runner uses uv to build isolated envs"
        )
    return found


def torch_index_for(accelerator: str, cuda_version: str | None = None) -> str | None:
    """Pick the torch wheel index for an accelerator.

    ``cuda_version`` (e.g. "12.4") selects a matching cuXXX index when given.
    """
    if accelerator != "cuda":
        return TORCH_INDEX_BY_ACCELERATOR.get(accelerator)
    if cuda_version:
        major, _, minor = cuda_version.partition(".")
        if major.isdigit():
            tag = f"cu{major}{(minor or '0')[:1]}"
            return f"https://download.pytorch.org/whl/{tag}"
    return TORCH_INDEX_BY_ACCELERATOR["cuda"]


def env_path(name: str) -> Path:
    return DIRS["envs"] / name


def python_path(name: str) -> Path:
    base = env_path(name)
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _fingerprint(packages: tuple[str, ...], index: str | None, overlay: tuple[str, ...] = ()) -> str:
    payload = {"packages": sorted(packages), "index": index}
    if overlay:  # keeps fingerprints of envs without an overlay unchanged
        payload["overlay"] = sorted(overlay)
    payload = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_marker(name: str) -> dict | None:
    marker = env_path(name) / _MARKER
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _run(argv: list[str], label: str) -> None:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise EnvProvisionError(
            f"{label} failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )


def ensure_env(
    name: str,
    packages: tuple[str, ...],
    accelerator: str = "cpu",
    cuda_version: str | None = None,
    python_version: str | None = None,
    overlay: tuple[str, ...] = (),
) -> Path:
    """Create (or reuse) an isolated env and return its interpreter path.

    ``overlay`` packages are force-reinstalled with --no-deps after everything
    else. It exists for one case: a library whose dependency check wants the
    GUI build of a package (opencv-contrib-python) while the host lacks the
    system libraries that build links against (libGL). The GUI dist stays
    installed for the metadata check; the headless build's files replace it.

    Idempotent: a marker file records the resolved package set and torch index,
    so a repeat call with the same inputs is a no-op.
    """
    index = torch_index_for(accelerator, cuda_version)
    want = _fingerprint(packages, index, overlay)

    marker = _read_marker(name)
    interpreter = python_path(name)
    if marker and marker.get("fingerprint") == want and interpreter.exists():
        return interpreter

    base = env_path(name)
    base.parent.mkdir(parents=True, exist_ok=True)

    uv = _uv()
    create = [uv, "venv", str(base)]
    if python_version:
        create += ["--python", python_version]
    _run(create, f"creating env {name!r}")

    # Torch first, from the accelerator-specific index. Installing it alongside
    # other packages lets an --extra-index-url resolution pull the PyPI (CUDA)
    # build instead, so the two steps must stay separate and ordered.
    torch_wanted = tuple(p for p in packages if p.split("=")[0].split("[")[0] in TORCH_PACKAGES)
    rest = tuple(p for p in packages if p not in torch_wanted)

    if torch_wanted:
        cmd = [uv, "pip", "install", "--python", str(interpreter), *torch_wanted]
        if index:
            cmd += ["--index-url", index]
        _run(cmd, f"installing torch into {name!r}")

    if rest:
        _run(
            [uv, "pip", "install", "--python", str(interpreter), *rest],
            f"installing packages into {name!r}",
        )

    if overlay:
        _run(
            [uv, "pip", "install", "--python", str(interpreter), "--reinstall", "--no-deps", *overlay],
            f"overlaying packages into {name!r}",
        )

    (base / _MARKER).write_text(
        json.dumps(
            {
                "fingerprint": want,
                "packages": list(packages),
                "torch_index": index,
                "accelerator": accelerator,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return interpreter


def run_in_env(
    name: str,
    argv: list[str],
    check: bool = True,
    extra_python_path: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with an isolated env's interpreter.

    ``extra_python_path`` makes vendored packages importable without
    installing them into the env.
    """
    interpreter = python_path(name)
    if not interpreter.exists():
        raise EnvProvisionError(f"env {name!r} has not been provisioned")

    env = os.environ.copy()
    if extra_python_path:
        env["PYTHONPATH"] = os.pathsep.join(
            [*extra_python_path, env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

    result = subprocess.run(
        [str(interpreter), *argv], capture_output=True, text=True, check=False, env=env
    )
    if check and result.returncode != 0:
        raise EnvProvisionError(
            f"command in env {name!r} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    return result


def env_bin(name: str, executable: str) -> Path:
    """Path to a console script installed inside an isolated env."""
    base = env_path(name)
    if os.name == "nt":
        return base / "Scripts" / f"{executable}.exe"
    return base / "bin" / executable
