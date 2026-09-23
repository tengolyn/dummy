"""Resolves, downloads and verifies llama.cpp release binaries.

The upstream Linux release is not a standalone static binary: it unpacks to a
directory of shared objects plus executables, needs its own directory on the
library search path, and depends on an OpenMP runtime the archive does not
ship. All three facts are handled here.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from indic_runner.config import DIRS

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases"

# Pinned because upstream's /releases/latest points at a tag that carries only
# a nightly-tag.txt, and because the CLI surface drifts between builds (-no-cnv
# was removed; one-shot completion moved from llama-cli to llama-completion).
# Override for maintenance only.
PINNED_BUILD = os.getenv("INDIC_RUNNER_LLAMA_BUILD", "b11118")

_BUILD_TAG = re.compile(r"^b\d+$")

# (os, arch, accelerator) -> release asset slug. Covers the full matrix the
# framework targets; a miss raises rather than guessing.
ASSET_MATRIX = {
    ("darwin", "arm64", "mps"): "macos-arm64",
    ("darwin", "arm64", "cpu"): "macos-arm64",
    ("darwin", "x86_64", "cpu"): "macos-x64",
    ("linux", "arm64", "cpu"): "ubuntu-arm64",
    ("linux", "arm64", "cuda"): "ubuntu-cuda-13.4-arm64",
    ("linux", "x86_64", "cpu"): "ubuntu-x64",
    ("linux", "x86_64", "cuda"): "ubuntu-cuda-12.8-x64",
    ("windows", "arm64", "cpu"): "win-cpu-arm64",
    ("windows", "x86_64", "cpu"): "win-cpu-x64",
    ("windows", "x86_64", "cuda"): "win-cuda-12.4-x64",
}

# OpenMP runtime each platform expects the host to provide, with the hint we
# surface when it is missing.
OPENMP_REQUIREMENT = {
    "linux": ("libgomp.so.1", "install it with your package manager, e.g. `apt-get install libgomp1`"),
    "darwin": (None, "macOS builds link Accelerate; no extra runtime needed"),
    "windows": (None, "Windows builds ship their own OpenMP runtime"),
}


class UnsupportedPlatform(RuntimeError):
    """Raised when no llama.cpp asset matches the host."""


class MissingRuntimeDependency(RuntimeError):
    """Raised when a downloaded binary cannot load its shared dependencies."""


@dataclass(frozen=True)
class BinaryBundle:
    """A resolved llama.cpp installation on disk."""

    root: Path
    build: str

    def executable(self, name: str) -> Path:
        suffix = ".exe" if os.name == "nt" else ""
        return self.root / f"{name}{suffix}"

    @property
    def library_dir(self) -> Path:
        # The executables resolve their .so/.dylib siblings from here; the
        # manifest carries this so `run` can set the loader path.
        return self.root


def asset_slug(host_os: str, arch: str, accelerator: str) -> str:
    """Look up the release asset slug for a host, or raise."""
    key = (host_os, arch, accelerator)
    if key in ASSET_MATRIX:
        return ASSET_MATRIX[key]
    # An accelerator without a dedicated build still runs on the CPU asset.
    fallback = (host_os, arch, "cpu")
    if fallback in ASSET_MATRIX:
        return ASSET_MATRIX[fallback]
    raise UnsupportedPlatform(
        f"no llama.cpp release asset for os={host_os!r} arch={arch!r} "
        f"accelerator={accelerator!r}; known combinations: "
        f"{sorted(ASSET_MATRIX)}"
    )


def asset_filename(build: str, slug: str) -> str:
    ext = "zip" if slug.startswith("win-") else "tar.gz"
    return f"llama-{build}-bin-{slug}.{ext}"


def resolve_build(fetch=urllib.request.urlopen) -> str:
    """Return the build tag to install.

    Uses the pin by default. When unpinned, lists releases and takes the first
    real ``b<digits>`` tag -- never ``/releases/latest``, which is unusable for
    this repository.
    """
    if PINNED_BUILD:
        return PINNED_BUILD
    with fetch(f"{RELEASES_URL}?per_page=20") as response:
        releases = json.loads(response.read().decode())
    for release in releases:
        tag = release.get("tag_name", "")
        if _BUILD_TAG.match(tag):
            return tag
    raise UnsupportedPlatform("no llama.cpp build tag found in the release listing")


def check_openmp(host_os: str) -> None:
    """Verify the platform's OpenMP runtime is loadable.

    The llama.cpp archives do not bundle it, so a missing runtime surfaces as
    an opaque loader error at first inference otherwise.
    """
    library, hint = OPENMP_REQUIREMENT.get(host_os, (None, ""))
    if library is None:
        return
    try:
        ctypes.CDLL(library)
    except OSError as exc:
        raise MissingRuntimeDependency(
            f"llama.cpp needs {library}, which is not installed: {hint}"
        ) from exc


def _extract(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(destination)


def _locate_root(extracted: Path) -> Path:
    """Find the directory actually holding the executables."""
    for candidate in [extracted, *extracted.iterdir()]:
        if not candidate.is_dir():
            continue
        for name in ("llama-cli", "llama-cli.exe", "llama-server", "llama-server.exe"):
            if (candidate / name).exists():
                return candidate
        nested = candidate / "build" / "bin"
        if nested.is_dir():
            return nested
    return extracted


def ensure_llama_cpp(
    host_os: str,
    arch: str,
    accelerator: str,
    build: str | None = None,
    download=urllib.request.urlretrieve,
) -> BinaryBundle:
    """Download and unpack llama.cpp for this host; idempotent."""
    build = build or resolve_build()
    slug = asset_slug(host_os, arch, accelerator)
    target = DIRS["bin"] / f"llama.cpp-{build}-{slug}"

    if not target.exists():
        filename = asset_filename(build, slug)
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{build}/{filename}"
        staging = DIRS["bin"] / f".staging-{build}-{slug}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        archive = staging / filename
        download(url, archive)
        _extract(archive, staging)
        archive.unlink(missing_ok=True)

        root = _locate_root(staging)
        root.rename(target) if root != staging else staging.rename(target)
        shutil.rmtree(staging, ignore_errors=True)

    for entry in target.iterdir():
        if entry.is_file() and not entry.suffix and os.name != "nt":
            entry.chmod(entry.stat().st_mode | 0o111)

    check_openmp(host_os)
    return BinaryBundle(root=target, build=build)


def current_platform() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else (
        "x86_64" if machine in ("x86_64", "amd64") else machine
    )
    return system, arch
