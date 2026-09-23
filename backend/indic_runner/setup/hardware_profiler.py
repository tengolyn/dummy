"""Detects OS, CUDA/MPS/CPU availability, and total/usable RAM.

Probing is split from parsing: every ``_parse_*`` function is a pure string ->
value mapping, so any platform's detection can be tested from captured fixture
output on any host.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass

# Fraction of physical RAM we are willing to hand to a model. The rest is
# headroom for the OS, the tokenizer, and the streaming pipeline's buffers.
RAM_HEADROOM_FACTOR = 0.75

_BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of the host's execution capabilities."""

    os: str  # "darwin" | "linux" | "windows"
    arch: str  # "arm64" | "x86_64" | ...
    accelerator: str  # "cuda" | "mps" | "cpu"
    total_ram_gb: float
    usable_ram_gb: float
    vram_gb: float | None
    gpu_name: str | None
    cuda_version: str | None
    cpu_threads: int

    @property
    def is_gpu(self) -> bool:
        return self.accelerator in ("cuda", "mps")

    @property
    def memory_budget_gb(self) -> float:
        """Memory the decision matrix may spend on weights.

        CUDA is bounded by VRAM; MPS shares system RAM with the OS, and CPU
        inference is bounded by system RAM.
        """
        if self.accelerator == "cuda" and self.vram_gb is not None:
            return self.vram_gb
        return self.usable_ram_gb


# --------------------------------------------------------------------------
# Pure parsers - each takes raw tool output and returns a value.
# --------------------------------------------------------------------------

def normalize_os(system: str) -> str:
    return {"darwin": "darwin", "linux": "linux", "windows": "windows"}.get(
        system.lower(), system.lower()
    )


def normalize_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("x86_64", "amd64"):
        return "x86_64"
    return m


def parse_nvidia_smi(output: str) -> tuple[str, float, str] | None:
    """Parse ``nvidia-smi --query-gpu=name,memory.total,driver_version``.

    Expects CSV with ``noheader,nounits``; memory is reported in MiB.
    Returns ``(gpu_name, vram_gb, driver_version)`` for the first GPU.
    """
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem_mib, driver = parts[0], parts[1], parts[2]
        try:
            vram_gb = float(mem_mib) / 1024
        except ValueError:
            continue
        return name, round(vram_gb, 2), driver
    return None


def parse_sysctl_memsize(output: str) -> float | None:
    """Parse ``sysctl -n hw.memsize`` (bytes) into GB."""
    text = output.strip()
    if not text.isdigit():
        return None
    return round(int(text) / _BYTES_PER_GB, 2)


def parse_meminfo(output: str) -> float | None:
    """Parse ``/proc/meminfo`` contents into total RAM in GB."""
    match = re.search(r"^MemTotal:\s+(\d+)\s*kB", output, re.MULTILINE)
    if not match:
        return None
    return round(int(match.group(1)) * 1024 / _BYTES_PER_GB, 2)


def parse_wmic_memory(output: str) -> float | None:
    """Parse Windows ``wmic ComputerSystem get TotalPhysicalMemory`` (bytes)."""
    for line in output.splitlines():
        text = line.strip()
        if text.isdigit():
            return round(int(text) / _BYTES_PER_GB, 2)
    return None


# --------------------------------------------------------------------------
# Probes - thin subprocess wrappers, each degrading to None on any failure.
# --------------------------------------------------------------------------

def _run(argv: list[str], timeout: float = 10.0) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def probe_cuda() -> tuple[str, float, str] | None:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return None
    return parse_nvidia_smi(output)


def probe_total_ram_gb(host_os: str) -> float | None:
    if host_os == "darwin":
        output = _run(["sysctl", "-n", "hw.memsize"])
        return parse_sysctl_memsize(output) if output else None
    if host_os == "linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                return parse_meminfo(handle.read())
        except OSError:
            return None
    if host_os == "windows":
        output = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"])
        return parse_wmic_memory(output) if output else None
    return None


def _fallback_ram_gb() -> float:
    """Last-resort RAM figure via sysconf, else a conservative 8GB."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / _BYTES_PER_GB, 2)
    except (AttributeError, ValueError, OSError):
        return 8.0


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def detect_accelerator(host_os: str, arch: str, cuda: tuple | None) -> str:
    if cuda is not None:
        return "cuda"
    if host_os == "darwin" and arch == "arm64":
        return "mps"
    return "cpu"


def profile_hardware() -> HardwareProfile:
    """Build a HardwareProfile for the current host.

    Never raises: any probe that fails degrades the profile toward ``cpu``.
    """
    host_os = normalize_os(platform.system())
    arch = normalize_arch(platform.machine())

    cuda = probe_cuda()
    accelerator = detect_accelerator(host_os, arch, cuda)

    total_ram = probe_total_ram_gb(host_os)
    if total_ram is None:
        total_ram = _fallback_ram_gb()

    gpu_name, vram_gb, cuda_version = (cuda if cuda else (None, None, None))

    return HardwareProfile(
        os=host_os,
        arch=arch,
        accelerator=accelerator,
        total_ram_gb=total_ram,
        usable_ram_gb=round(total_ram * RAM_HEADROOM_FACTOR, 2),
        vram_gb=vram_gb,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        cpu_threads=os.cpu_count() or 1,
    )
