#!/usr/bin/env bash
# One-shot environment setup for Indic-Runner Phase 1.
#
# Idempotent: safe to re-run. Verifies rather than assumes, and stops at the
# first thing that would only fail later during a multi-GB download.
#
#   ./bootstrap.sh                 # set up and verify
#   ./bootstrap.sh --smoke         # also run a real 2.5B setup end to end
#
# INDIC_RUNNER_HOME controls where models, binaries and manifests live.
# On a cloud VM point it at local SSD (e.g. /mnt/indic-runner), never at a
# network share: conversion writes tens of GB.

set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BACKEND_DIR"

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# --- 1. uv ------------------------------------------------------------------
step "uv"
if ! command -v uv >/dev/null 2>&1; then
    warn "not found, installing"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
        || die "uv install failed; see https://docs.astral.sh/uv/"
    for candidate in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [[ -x "$candidate/uv" ]] && export PATH="$candidate:$PATH"
    done
fi
command -v uv >/dev/null 2>&1 || die "uv still not on PATH; open a new shell and re-run"
ok "$(uv --version)"

# --- 2. OpenMP runtime ------------------------------------------------------
# llama.cpp's release binaries are not self-contained and fail to load without
# it. Checked here because the error surfaces only at first inference.
step "OpenMP runtime"
case "$(uname -s)" in
  Linux)
    if ldconfig -p 2>/dev/null | grep -q 'libgomp\.so\.1'; then
        ok "libgomp present"
    elif command -v apt-get >/dev/null 2>&1; then
        warn "libgomp missing, installing"
        sudo apt-get update -qq && sudo apt-get install -y -qq libgomp1 \
            || die "could not install libgomp1; run: sudo apt-get install libgomp1"
        ok "libgomp installed"
    else
        die "libgomp.so.1 missing; install your distro's libgomp package"
    fi
    ;;
  Darwin) ok "macOS links Accelerate; nothing to install" ;;
  *)      warn "unrecognised OS $(uname -s); skipping OpenMP check" ;;
esac

# --- 3. Python environment --------------------------------------------------
step "Python environment"
uv sync --quiet || die "uv sync failed"
ok "$(uv run python --version) with dependencies installed"

# --- 4. Credentials ---------------------------------------------------------
step "Hugging Face credentials"
if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN set in the environment"
elif [[ -f .env ]] && grep -qE '^HF_TOKEN=.+' .env; then
    ok "HF_TOKEN found in .env"
else
    warn "no token configured"
    echo "      Gated models (Airavata, IndicTrans2) will be refused without one."
    echo "      Create a read token at https://huggingface.co/settings/tokens, then:"
    echo "        echo 'HF_TOKEN=hf_xxx' > $BACKEND_DIR/.env"
    [[ -f .env ]] || cp .env.example .env 2>/dev/null || true
fi

# --- 5. Storage -------------------------------------------------------------
# Conversion peaks at roughly 4.7x the final artifact: for an 8.5B model that
# is ~39GB (safetensors + f16 intermediate + quantized output) before the
# source is evicted.
step "Storage"
HOME_DIR="${INDIC_RUNNER_HOME:-$HOME/.indic-runner}"
mkdir -p "$HOME_DIR"
AVAIL_GB=$(df -BG --output=avail "$HOME_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
ok "INDIC_RUNNER_HOME=$HOME_DIR"
if [[ "$AVAIL_GB" -eq 0 ]]; then
    warn "could not determine free space"
elif [[ "$AVAIL_GB" -lt 40 ]]; then
    warn "${AVAIL_GB}GB free — one 8.5B conversion peaks near 39GB"
elif [[ "$AVAIL_GB" -lt 100 ]]; then
    ok "${AVAIL_GB}GB free (enough per-model; ~100GB for the full roster)"
else
    ok "${AVAIL_GB}GB free"
fi
case "$HOME_DIR" in
  */cloudfiles/*) warn "this looks like a network share — conversion will be slow; prefer /mnt" ;;
esac

# --- 6. Verify --------------------------------------------------------------
step "Test suite"
uv run pytest tests -q || die "tests failed — do not proceed"
ok "all tests passed"

step "Platform resolution"
uv run python - <<'PY' || die "platform probe failed"
from indic_runner.setup.binary_manager import asset_filename, asset_slug, current_platform
from indic_runner.setup.hardware_profiler import profile_hardware

hw = profile_hardware()
print(f"  hardware   {hw.os}/{hw.arch} {hw.accelerator}, "
      f"{hw.cpu_threads} threads, {hw.total_ram_gb}GB RAM")
if hw.gpu_name:
    print(f"  gpu        {hw.gpu_name} ({hw.vram_gb}GB, CUDA {hw.cuda_version})")
os_name, arch = current_platform()
print(f"  llama.cpp  {asset_filename('b11118', asset_slug(os_name, arch, hw.accelerator))}")
PY

# --- 7. Optional smoke test -------------------------------------------------
if [[ "$SMOKE" -eq 1 ]]; then
    step "Smoke test (real setup: sarvam-1, 2.5B)"
    uv run indic-runner setup sarvam-1 --dry-run || die "dry run failed"
    uv run indic-runner setup sarvam-1 --force || die "setup failed"
    ok "manifest written to $HOME_DIR/manifests/sarvam-1.json"
fi

step "Ready"
cat <<EOF
  export INDIC_RUNNER_HOME=$HOME_DIR

  uv run indic-runner setup <alias> --dry-run   # resolve the plan, download nothing
  uv run indic-runner setup <alias>             # build the artifact + manifest

  Start small and step up: sarvam-1 (2.5B) -> airavata-7b (6.9B) -> navarasa-2.0-7b (8.5B)
EOF
