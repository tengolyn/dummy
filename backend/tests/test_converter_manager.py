"""Vendoring llama.cpp's conversion tree."""

from __future__ import annotations

import io
import tarfile

import pytest

from indic_runner.setup import converter_manager as cm


def _fake_source_tarball(path, build="b11118", include=cm.VENDORED_PATHS):
    """Build a tarball shaped like a llama.cpp source release."""
    top = f"llama.cpp-{build}"
    with tarfile.open(path, "w:gz") as tf:
        for name in include:
            if name.endswith(".py"):
                data = b"# converter\n"
                info = tarfile.TarInfo(f"{top}/{name}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                data = b"x"
                info = tarfile.TarInfo(f"{top}/{name}/__init__.py")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        # Bulk of the real tree that must not be vendored.
        extra = tarfile.TarInfo(f"{top}/src/llama.cpp")
        extra.size = 1
        tf.addfile(extra, io.BytesIO(b"x"))


def test_vendors_only_the_conversion_paths(tmp_path, monkeypatch):
    """The source release is ~36MB; only ~2MB of it is the converter."""
    monkeypatch.setitem(cm.DIRS, "bin", tmp_path / "bin")

    def download(url, destination):
        _fake_source_tarball(destination)

    tree = cm.ensure_converter("b11118", download=download)
    assert tree.script.exists()
    assert (tree.root / "conversion").exists()
    assert (tree.root / "gguf-py").exists()
    assert not (tree.root / "src").exists()  # the rest is left behind


def test_python_path_covers_conversion_and_gguf(tmp_path, monkeypatch):
    monkeypatch.setitem(cm.DIRS, "bin", tmp_path / "bin")
    tree = cm.ConverterTree(root=tmp_path / "vendor", build="b11118")
    entries = tree.python_path
    assert str(tree.root) in entries          # `conversion` resolves from root
    assert str(tree.root / "gguf-py") in entries  # `gguf` lives here


def test_second_call_does_not_redownload(tmp_path, monkeypatch):
    monkeypatch.setitem(cm.DIRS, "bin", tmp_path / "bin")
    calls = []

    def download(url, destination):
        calls.append(url)
        _fake_source_tarball(destination)

    cm.ensure_converter("b11118", download=download)
    cm.ensure_converter("b11118", download=download)
    assert len(calls) == 1


def test_partial_tree_is_rebuilt(tmp_path, monkeypatch):
    """An interrupted fetch must not leave a tree that fails confusingly later."""
    monkeypatch.setitem(cm.DIRS, "bin", tmp_path / "bin")
    root = cm.converter_dir("b11118")
    root.mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# stale, incomplete\n")

    def download(url, destination):
        _fake_source_tarball(destination)

    tree = cm.ensure_converter("b11118", download=download)
    assert (tree.root / "conversion").exists()


def test_tarball_without_the_converter_is_reported(tmp_path, monkeypatch):
    monkeypatch.setitem(cm.DIRS, "bin", tmp_path / "bin")

    def download(url, destination):
        _fake_source_tarball(destination, include=())

    with pytest.raises(cm.ConverterUnavailable):
        cm.ensure_converter("b11118", download=download)


def test_download_failure_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setitem(cm.DIRS, "bin", tmp_path / "bin")

    def download(url, destination):
        raise OSError("network unreachable")

    with pytest.raises(cm.ConverterUnavailable, match="could not download"):
        cm.ensure_converter("b11118", download=download)


def test_url_targets_the_pinned_tag():
    assert "{build}" in cm.SOURCE_URL
    assert cm.SOURCE_URL.format(build="b11118").endswith("b11118.tar.gz")
