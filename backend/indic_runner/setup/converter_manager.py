"""Vendors llama.cpp's HF->GGUF conversion tree at a pinned tag.

The release binaries do not ship the converter, and it is no longer a single
standalone script: ``convert_hf_to_gguf.py`` imports a ``conversion`` package
(one module per architecture) plus ``gguf-py``. Vendoring all three lets setup
build every GGUF from official weights with identical settings, rather than
depending on third-party quantizers whose imatrix and base revision are
unknown -- which would otherwise confound any comparison between models.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from indic_runner.config import DIRS
from indic_runner.setup.binary_manager import PINNED_BUILD

SOURCE_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{build}.tar.gz"

# Only these paths are kept; the rest of the 36MB source tree is not needed.
VENDORED_PATHS = ("convert_hf_to_gguf.py", "conversion", "gguf-py")

CONVERT_SCRIPT = "convert_hf_to_gguf.py"


class ConverterUnavailable(RuntimeError):
    """Raised when the conversion tree could not be vendored."""


@dataclass(frozen=True)
class ConverterTree:
    """A vendored conversion tree on disk."""

    root: Path
    build: str

    @property
    def script(self) -> Path:
        return self.root / CONVERT_SCRIPT

    @property
    def python_path(self) -> list[str]:
        """Entries the converter needs on PYTHONPATH.

        ``conversion`` resolves from the root; ``gguf`` lives under gguf-py.
        """
        return [str(self.root), str(self.root / "gguf-py")]


def converter_dir(build: str) -> Path:
    return DIRS["bin"] / f"llama-converter-{build}"


def _is_complete(root: Path) -> bool:
    return all((root / name).exists() for name in VENDORED_PATHS)


def _extract_vendored(archive: Path, destination: Path) -> None:
    """Unpack only the conversion paths from the release tarball."""
    with tarfile.open(archive) as tf:
        top = tf.getnames()[0].split("/")[0]
        wanted = tuple(f"{top}/{name}" for name in VENDORED_PATHS)
        members = [m for m in tf.getmembers() if m.name.startswith(wanted)]
        if not members:
            raise ConverterUnavailable(
                f"llama.cpp source at this tag has none of {VENDORED_PATHS}"
            )
        with tempfile.TemporaryDirectory() as staging:
            tf.extractall(staging, members=members)
            extracted = Path(staging) / top
            destination.mkdir(parents=True, exist_ok=True)
            for name in VENDORED_PATHS:
                source = extracted / name
                if source.exists():
                    shutil.move(str(source), str(destination / name))


def ensure_converter(
    build: str | None = None,
    download=urllib.request.urlretrieve,
) -> ConverterTree:
    """Fetch and unpack the conversion tree; idempotent."""
    build = build or PINNED_BUILD
    root = converter_dir(build)

    if _is_complete(root):
        return ConverterTree(root=root, build=build)

    # A partial tree from an interrupted fetch would fail confusingly later.
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        archive = Path(staging) / f"{build}.tar.gz"
        try:
            download(SOURCE_URL.format(build=build), archive)
        except OSError as exc:
            raise ConverterUnavailable(
                f"could not download the llama.cpp source for {build}: {exc}"
            ) from exc
        _extract_vendored(archive, root)

    if not _is_complete(root):
        raise ConverterUnavailable(f"conversion tree at {root} is incomplete")
    return ConverterTree(root=root, build=build)
