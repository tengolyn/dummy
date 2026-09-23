"""Parser tests driven by captured tool output, so every OS is covered here."""

from __future__ import annotations

import pytest
from helpers import make_hw

from indic_runner.setup import hardware_profiler as hp

NVIDIA_SMI_SINGLE = "NVIDIA A100-SXM4-80GB, 81920, 550.54.15\n"
NVIDIA_SMI_MULTI = (
    "NVIDIA GeForce RTX 3070, 8192, 535.129.03\n"
    "NVIDIA GeForce RTX 3070, 8192, 535.129.03\n"
)
SYSCTL_MEMSIZE = "34359738368\n"  # 32 GiB, macOS
PROC_MEMINFO = """MemTotal:       16318492 kB
MemFree:         2417588 kB
MemAvailable:    9876543 kB
"""
WMIC_MEMORY = "TotalPhysicalMemory\n17179869184\n\n"


@pytest.mark.parametrize(
    "machine,expected",
    [("arm64", "arm64"), ("aarch64", "arm64"), ("x86_64", "x86_64"),
     ("AMD64", "x86_64"), ("riscv64", "riscv64")],
)
def test_normalize_arch(machine, expected):
    assert hp.normalize_arch(machine) == expected


@pytest.mark.parametrize(
    "system,expected",
    [("Darwin", "darwin"), ("Linux", "linux"), ("Windows", "windows")],
)
def test_normalize_os(system, expected):
    assert hp.normalize_os(system) == expected


def test_parse_nvidia_smi_single_gpu():
    name, vram, driver = hp.parse_nvidia_smi(NVIDIA_SMI_SINGLE)
    assert name == "NVIDIA A100-SXM4-80GB"
    assert vram == 80.0
    assert driver == "550.54.15"


def test_parse_nvidia_smi_takes_first_of_several():
    name, vram, _ = hp.parse_nvidia_smi(NVIDIA_SMI_MULTI)
    assert name == "NVIDIA GeForce RTX 3070"
    assert vram == 8.0


@pytest.mark.parametrize("junk", ["", "\n", "no GPU found", "name, notanumber, drv"])
def test_parse_nvidia_smi_rejects_junk(junk):
    assert hp.parse_nvidia_smi(junk) is None


def test_parse_sysctl_memsize():
    assert hp.parse_sysctl_memsize(SYSCTL_MEMSIZE) == 32.0


def test_parse_sysctl_memsize_rejects_non_numeric():
    assert hp.parse_sysctl_memsize("hw.memsize: unavailable") is None


def test_parse_meminfo():
    assert hp.parse_meminfo(PROC_MEMINFO) == pytest.approx(15.56, abs=0.01)


def test_parse_meminfo_without_memtotal():
    assert hp.parse_meminfo("MemFree: 100 kB\n") is None


def test_parse_wmic_memory():
    assert hp.parse_wmic_memory(WMIC_MEMORY) == 16.0


@pytest.mark.parametrize(
    "host_os,arch,cuda,expected",
    [
        ("linux", "x86_64", ("A100", 80.0, "12.4"), "cuda"),
        ("darwin", "arm64", ("A100", 80.0, "12.4"), "cuda"),  # CUDA wins if present
        ("darwin", "arm64", None, "mps"),
        ("darwin", "x86_64", None, "cpu"),  # Intel Macs have no MPS
        ("linux", "arm64", None, "cpu"),
        ("windows", "x86_64", None, "cpu"),
    ],
)
def test_detect_accelerator(host_os, arch, cuda, expected):
    assert hp.detect_accelerator(host_os, arch, cuda) == expected


def test_memory_budget_uses_vram_on_cuda():
    cuda = make_hw("cuda", total_ram_gb=32.0, vram_gb=24.0)
    assert cuda.memory_budget_gb == 24.0


@pytest.mark.parametrize("accelerator", ["cpu", "mps"])
def test_memory_budget_uses_system_ram_without_vram(accelerator):
    hw = make_hw(accelerator, total_ram_gb=32.0)
    assert hw.memory_budget_gb == 24.0  # 75% headroom


def test_profile_hardware_never_raises_on_this_host():
    profile = hp.profile_hardware()
    assert profile.accelerator in ("cuda", "mps", "cpu")
    assert profile.total_ram_gb > 0
    assert profile.cpu_threads >= 1
