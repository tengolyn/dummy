# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Phase 1 (`setup`, the AOT compiler) is implemented and tested** under `backend/indic_runner/setup/`.
Phase 2 (`run`, the streaming VM) is implemented under `backend/indic_runner/runtime/`: llama.cpp and vLLM daemons plus stdio workers for ctranslate2/transformers, tested against fakes only. **No OCR worker yet** (surya/paddleocr/easyocr/transformers-OCR raise a clear `EngineError`), and no real-model run has been done. Run: `indic-runner run --model <alias> --dataset x.jsonl` (OCR needs `--ocr-variant`); output goes to `runs/<run_id>/results.json`; `--resume <run_id>` continues.
`docs/architecture.md` is the design source of truth, with two corrections noted below.

## Commands

`./backend/bootstrap.sh` does first-time setup on a fresh machine: installs uv, checks the OpenMP
runtime llama.cpp needs, syncs dependencies, checks credentials and free space, runs the tests, and
prints the resolved hardware lane and llama.cpp asset. Idempotent. `--smoke` additionally runs a real
2.5B setup end to end.

Run all commands from `backend/`. Package management is **uv** (not pip/poetry) — `pyproject.toml` + `uv.lock` live in `backend/`.

- `uv sync` — install/update the environment from the lockfile (creates `backend/.venv/`)
- `uv run pytest tests -q` — run tests
- `uv run indic-runner --help` — run the CLI entrypoint (`setup` / `run` subcommands)
- `uv add <pkg>` / `uv add --dev <pkg>` — add a runtime / dev dependency
- `uv run pytest tests/test_decision_matrix.py -q` — run one suite
- `uv run indic-runner setup <alias-or-repo> --task <task> --dry-run` — resolve a plan and print the
  manifest without downloading anything. The fastest way to check tiering logic.
- `INDIC_RUNNER_HOME=/tmp/x uv run indic-runner setup ...` — redirect framework state for a scratch run.

### Credentials

`HF_TOKEN` goes in `backend/.env` (gitignored; copy `backend/.env.example`). `indic_runner/config.py`
loads it at import, so `hf_token()` is available everywhere with no wiring — it reaches both the
profiler's `HfApi` and `snapshot_download`. Real environment variables take precedence over the file,
and empty values are skipped so a copied blank placeholder cannot mask a working token. Never print
the token: `cli._auth_line()` reports only whether one is configured and from which file.

### Third correction: "in-process" engines are stdio workers

The orchestrator cannot import torch/CTranslate2 (isolated envs), so in-process engines run as one long-lived worker under the env's interpreter, speaking JSON lines over stdio (`runtime/engines/worker_engine.py`, `runtime/workers/`). The env name is `manifest.engine.binary_or_env` (setup stores the engine name there).

### Two corrections to docs/architecture.md, already reflected in the code

1. **llama.cpp is not a standalone static binary** (Decision 3). The Linux release unpacks to ~35 `.so`
   files, needs its own directory on `LD_LIBRARY_PATH`, and depends on an unbundled `libgomp.so.1`.
   `binary_manager` preflights it; the manifest carries `engine.library_dir` for this reason.
2. **OCR is "never low-bit", not "locked to FP16/BF16"**. fp16 is not a real CPU dtype in PyTorch, so
   the OCR lane is fp16 on GPU/MPS and **bf16 on CPU** — bf16 keeps fp32's exponent range (only the
   mantissa narrows), so visual resolution survives at half the memory. That halving is what lets a
   5B-class OCR model such as Chandra run on a 32GB machine at all; at fp32 it needed ~25GB and setup
   correctly refused.

## What Indic-Runner is

A model- and hardware-agnostic CLI execution framework for Indic-language AI workloads (reasoning, summarization, translation, OCR), architected as an **Ahead-of-Time (AOT) compiler** + **Streaming Virtual Machine** rather than a monolithic script. Core philosophy: pay setup cost once, run with zero overhead during execution.

## Architecture (from docs/architecture.md)

Two distinct phases:

1. **`setup <hf_repo>`** — AOT Compiler phase: Hardware Profiler (OS/CUDA/MPS/CPU/RAM) → Model Profiler (HF API architecture/params) → Decision Matrix (task + hardware + precision → engine selection) → Artifact Compiler (download, quantize/convert, evict raw FP16 weights). Emits a declarative `manifest.json` execution contract.

2. **`run --model ...`** — Streaming VM phase: Manifest Loader reads `manifests/<alias>.json` → Engine Lifecycle Manager launches either a daemon (vLLM/llama-server, subprocess + healthcheck) or in-process engine (CTranslate2/OCR, direct memory load) → Streaming Pipeline (chunked reader, batch dispatcher) → Task Adapter (per task type) → Output Sink (streams to `results.jsonl`, auto teardown).

This project is a **runner only** — it executes inference and streams results to disk. It does not compute or report evaluation metrics (no BLEU/CER/ROUGE scoring). Any accuracy/quality evaluation happens downstream, outside this codebase.

### Key design decisions to preserve

- **Hardware-aware precision tiering**, decided before any weights download:
  - OCR: always FP16/BF16 (quantization destroys visual feature resolution needed for matras/scripts)
  - Translation: INT8 (CTranslate2) or GGUF Q4 (decoder-based translators like Sarvam-Translate)
  - LLM (reasoning/summarization): GGUF Q4_K_M on Apple Silicon/CPU; FP16/BF16 on NVIDIA with ample VRAM (vLLM PagedAttention/continuous batching); AWQ/FP8 on constrained VRAM
- **AOT weight compilation**: if the optimal artifact isn't on HF Hub, `setup` compiles it locally (download safetensors → isolated compiler → convert → delete source FP16). Saves ~50% disk, keeps `run` instant.
- **Isolated runtimes to avoid dependency conflicts**: llama.cpp is a pre-compiled standalone static binary; vLLM and CTranslate2 each live in isolated, hidden micro-virtualenvs triggered dynamically. Never merge these dependency stacks into one environment.
- **Streaming only**: `run` must never hold a full dataset or output array in memory — datasets (JSONL/Parquet/image dirs) are read in sequential chunks, outputs appended to disk as inference yields them. Only lightweight runtime telemetry (tokens/sec, throughput) belongs here — no quality/evaluation metrics.
- **Hardcoded, immutable paths** — no path-resolution heuristics. All framework state lives under `BASE_DIR` (default `~/.indic-runner`, overridable via `INDIC_RUNNER_HOME` env var) in `indic_runner/config.py`, with fixed subdirs: `bin/` (static binaries), `envs/` (isolated venvs), `models/` (compiled weights), `manifests/` (JSON execution contracts). The CLI itself only ever accepts model aliases and dataset paths — do not introduce new user-facing path configuration.

### Artifact provenance

`manifest.artifact_source` records the repo the weights actually came from and whether it is
`official` (the model's own org), `third-party` (a community quantizer) or `converted` (built
locally). Without it a run is unattributable — you cannot tell which GGUF build produced a result.
`GgufCompiler` also verifies the fetched file matches the planned quantization and raises
`QuantizationMismatch` otherwise: silently accepting a Q5 build under a manifest claiming `q4_k_m`
would make the manifest lie about what ran.

### The manifest contract

`manifest.json` (see example at `docs/architecture.md:168-193`) is the sole interface between `setup` and `run` — `run` must execute strictly from this file with no internet access or heuristic guessing. Fields: `manifest_version`, `model_alias`, `base_repo`, `task`, `target_hardware`, `precision`, `engine` (`name`/`mode`/`binary_or_env`), `paths` (`artifacts_dir`/`tokenizer_dir`), `runtime_parameters` (`max_batch_size`/`context_length`).

### Supported models (exactly 4 per task, sub-10B params)

- **Reasoning**: Qwen2.5-7B-Instruct, Qwen3-8B, Phi-4-mini-instruct (3.8B), Sarvam-1 (2B)
- **Summarization**: Navarasa 2.0 (7B), Airavata (7B), Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507 (262k ctx)
- **Translation** (5 — see below): IndicTrans2 en-indic (1.1B), IndicTrans2 indic-en (1.1B),
  NLLB-200 (3.3B), MADLAD-400 (7.2B), Sarvam-Translate (4B)
- **OCR**: Indic-OCR by Bodhan AI (0.8B), Surya OCR (~650M), PaddleOCR PP-OCRv4 (<100M),
  Chandra OCR (5.3B)

**Every GGUF is built here, from the model's official safetensors.** No published GGUF is used —
not even one an org ships for its own model (`GgufCompiler.USE_PUBLISHED_GGUF = False`). A build made
by the model's authors may be imatrix-calibrated with settings we cannot see, so sourcing some models
that way and converting the rest would make part of every result a property of the quantizer.
Converting everything through one pipeline leaves a constant offset that cancels in comparison,
instead of a bias correlated with which orgs publish GGUFs — well-resourced labs do, the
Indic-specific models mostly do not, so that bias would run against the models this project exists to
evaluate. `resolve_target` also raises `ThirdPartyWeights` if a `gguf_repo` names another org, before
any download.

Known cost: all models sit slightly below their best-available form, since plain Q4_K_M is weaker
than an imatrix-calibrated build. The fix is uniform imatrix conversion from a fixed Indic
calibration corpus — controlled *and* high quality — but the corpus choice is its own methodological
decision, so it belongs after Phase 2.

**Local GGUF conversion works** via `setup/converter_manager.py`, which vendors llama.cpp's
conversion tree (`convert_hf_to_gguf.py` + `conversion/` + `gguf-py/`, ~2MB) from the source tarball
at the pinned build tag. The release *binaries* do not ship the converter, and the upstream script
imports a `conversion` package the `gguf` PyPI package does not provide — hence vendoring rather
than pip-installing. The vendored tree goes on `PYTHONPATH` via `run_in_env(extra_python_path=...)`.

Four per task, **except translation, which carries five**: IndicTrans2 publishes one model per
direction and gates each separately, so en-indic and indic-en are distinct entries. Stick to this
roster rather than adding others ad hoc.

### Registry realities (validated against the live Hub)

- **No model in the roster is `manual`-gated.** Llama-3.1-8B (×2 slots) and Gemma-2-9B were replaced
  with Qwen3-8B, Phi-4-mini and Qwen3-4B-Instruct-2507 precisely because manual gating blocks setup on
  a third party's approval queue. Keep it that way; licences are permissive (Apache-2.0 / MIT), so
  non-commercial weights such as Aya Expanse (CC-BY-NC) are excluded too.
- **Gating is per-repo, and for IndicTrans2 per-direction**: accepting `en-indic` does not grant
  `indic-en` or `indic-indic`. Airavata and both IndicTrans2 directions are `auto`-gated.
  `GatedRepository.gating` carries the type, since `auto` and `manual` need different user action.
  Never swallow the gating error: without the config the profiler cannot see the architecture, so the
  matrix would pick a plausible-but-wrong engine and only fail later, at download time.
- **EasyOCR does not publish to the Hub at all** (`repo: None`); it fetches weights from its own CDN
  at first use. `setup` raises `NotOnHub`; it is provisioned with the OCR runtime env instead.
- **The four OCR models are four different stacks**, so each registry entry declares a `runtime`
  (`transformers`, `surya`, `paddleocr`, `easyocr`). PaddleOCR's config declares no `architectures`
  at all — do not assume OCR means transformers.
- Parameter counts come from safetensors metadata and can differ from the roster's nominal size
  (Navarasa "7B" is 8.54B; MADLAD "7.2B" is 8.3B). Metadata wins.

### Input dataset schemas (per task)

Each task's input dataset (JSONL) has a fixed, task-specific column set — validate against these before streaming, don't infer columns:

- **Translation**: `id`, `language`, `target_language`, `reference`, `input`
- **OCR**: `id`, `image_path`, `language`, `reference`
- **Reasoning**: `id`, `input`, `language`, `reference`, `judge_rubric`
- **Summarization**: `id`, `language`, `input`, `reference`

`reference` is carried through for downstream evaluation only — this runner never scores against it.

### Output format (run results)

`run` emits a single JSON object with two top-level keys: `manifest` (run metadata) and `results` (array of per-row outputs). This is distinct from the `setup`-phase `manifest.json` execution contract described above — this one is a record of what happened during a `run`, written after the run.

- **`manifest`** — `run_id`, `model` (`name`/`revision`/`backend`/`path`), `dataset` (`name`/`use_case`/`language`/`languages`/`row_count`/`ocr_variant` — `"normal"` | `"scanned"` | `null` when `use_case` is not OCR), `runtime` (`device_requested`/`device_used`/`hardware`/`quantisation`/`dtype`/`fallback_reason`/`latency_p50_s`/`latency_p95_s`/`throughput_rows_per_s`/`peak_vram_gb`/`performance_notes`/`timeout_s`/`max_retries`/`resumed`/`python_version`), `generation` (`batch_size`/`max_new_tokens`/`temperature`/`do_sample`/`num_beams`), `run` (`created_at`/`status`/`successful_count`/`failed_count`/`resumed`).
- **`results[]`** — one entry per dataset row: `id`, `input`, `response`, `reasoning`, `latency_s`, `status`, `error` (nullable), `image_sha256` (nullable, OCR only).

Because streaming must never hold the full output array in memory, `results` is accumulated on disk as it streams (e.g. append-only JSONL per row) and only assembled into this final `manifest` + `results` JSON shape at the end of the run — do not build this structure in memory row-by-row during a large run.
