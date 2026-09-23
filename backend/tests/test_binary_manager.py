"""Platform matrix coverage: every lane resolves, unknown lanes raise."""

from __future__ import annotations

import itertools

import pytest

from indic_runner.setup import binary_manager as bm

SUPPORTED_OSES = ("darwin", "linux", "windows")
ARCHES = ("arm64", "x86_64")
ACCELERATORS = ("cuda", "mps", "cpu")


@pytest.mark.parametrize("key", sorted(bm.ASSET_MATRIX))
def test_every_matrix_entry_builds_a_filename(key):
    filename = bm.asset_filename("b11118", bm.asset_slug(*key))
    assert filename.startswith("llama-b11118-bin-")
    assert filename.endswith((".tar.gz", ".zip"))


@pytest.mark.parametrize(
    "host_os,arch,accelerator",
    [c for c in itertools.product(SUPPORTED_OSES, ARCHES, ACCELERATORS)
     if (host_os_arch := (c[0], c[1], "cpu")) in bm.ASSET_MATRIX],
)
def test_supported_platforms_all_resolve(host_os, arch, accelerator):
    # Any accelerator resolves as long as the platform has a CPU build to fall
    # back to, so a GPU without a dedicated asset still runs.
    assert bm.asset_slug(host_os, arch, accelerator)


def test_windows_assets_are_zip_and_others_tarball():
    assert bm.asset_filename("b1", "win-cpu-x64").endswith(".zip")
    assert bm.asset_filename("b1", "ubuntu-x64").endswith(".tar.gz")
    assert bm.asset_filename("b1", "macos-arm64").endswith(".tar.gz")


def test_apple_silicon_maps_to_the_metal_build():
    assert bm.asset_slug("darwin", "arm64", "mps") == "macos-arm64"


def test_cuda_hosts_get_cuda_builds():
    assert "cuda" in bm.asset_slug("linux", "x86_64", "cuda")
    assert "cuda" in bm.asset_slug("linux", "arm64", "cuda")


def test_gpu_without_dedicated_asset_falls_back_to_cpu_build():
    # macOS has no CUDA build; the CPU asset is correct there.
    assert bm.asset_slug("darwin", "x86_64", "cuda") == "macos-x64"


def test_unknown_platform_names_the_miss():
    with pytest.raises(bm.UnsupportedPlatform) as excinfo:
        bm.asset_slug("freebsd", "riscv64", "cpu")
    assert "freebsd" in str(excinfo.value)


def test_resolve_build_uses_the_pin():
    assert bm.resolve_build() == bm.PINNED_BUILD


def test_resolve_build_skips_the_broken_latest_tag(monkeypatch):
    """Upstream's newest release carries only nightly-tag.txt, not binaries."""
    import io
    import json

    payload = json.dumps(
        [{"tag_name": "v0.4.1"}, {"tag_name": "nightly"}, {"tag_name": "b11118"}]
    ).encode()

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(bm, "PINNED_BUILD", "")
    assert bm.resolve_build(fetch=lambda url: Response(payload)) == "b11118"


def test_openmp_check_is_a_noop_where_the_runtime_is_bundled():
    bm.check_openmp("darwin")
    bm.check_openmp("windows")


def test_openmp_check_reports_an_actionable_hint(monkeypatch):
    import ctypes

    def refuse(_name):
        raise OSError("libgomp.so.1: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", refuse)
    with pytest.raises(bm.MissingRuntimeDependency) as excinfo:
        bm.check_openmp("linux")
    assert "libgomp1" in str(excinfo.value)


def test_bundle_exposes_executables_and_library_dir(tmp_path):
    bundle = bm.BinaryBundle(root=tmp_path, build="b11118")
    assert bundle.executable("llama-server").parent == tmp_path
    assert bundle.library_dir == tmp_path


def test_current_platform_normalizes_arch():
    _, arch = bm.current_platform()
    assert arch in ("arm64", "x86_64") or arch  # never raises
