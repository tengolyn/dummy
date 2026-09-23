Here is the finalized system architecture and design document, updated to strictly feature 4 top-tier models per task, including the addition of **Sarvam-Translate** to the translation pipeline.

```markdown
# Indic-Runner: System Architecture & Design Decisions

## 1. Overview & Core Philosophy
**Indic-Runner** is a highly efficient, model- and hardware-agnostic execution framework explicitly designed for Indic AI workloads. 

Instead of operating as a traditional monolithic Python script, it is architected as an **Ahead-of-Time (AOT) compiler** and a **Streaming Virtual Machine**. The core philosophy is: **Pay the cost once during setup; run with zero overhead during execution.**

---

## 2. Target Workloads & Verified Models (Sub-10B Parameters)
The framework strictly targets four distinct use cases, utilizing exactly the top 4 most efficient open-weight models available for Indian languages per task:

### 1. Reasoning (Logical Structuring & Q&A)
1. **Qwen2.5-7B-Instruct:** SOTA benchmark performer with robust multilingual pre-training.
2. **Llama-3.1-8B-Instruct:** Meta's natively strong multilingual model, foundational for Indic reasoning and agentic tasks.
3. **Gemma-2-9B-It:** Google's open-weights model featuring dense reasoning capabilities and a highly efficient vocabulary.
4. **Sarvam-1 (2B):** Sarvam AI's highly efficient, purpose-built model utilizing a custom Indic tokenizer for ultra-fast, lightweight reasoning.

### 2. Summarization (Long-Context & Document Processing)
1. **Navarasa 2.0 (7B):** Gemma 7B-based, instruction-tuned on Indic translated Alpaca datasets for native RAG and summarization.
2. **Airavata (7B):** Open-source instruction-tuned model explicitly fine-tuned for Indic NLP tasks and summarization instructions.
3. **Qwen2.5-7B-Instruct:** Features a 128k context window, making it highly effective for summarizing massive regional documents without context loss.
4. **Llama-3.1-8B-Instruct:** Also features a 128k context window, frequently utilized alongside the Ambari framework for long-form native-language summarization.

### 3. Translation (Seq2Seq & Document Translation)
1. **IndicTrans2 (1.1B):** AI4Bharat's state-of-the-art model natively supporting all 22 scheduled Indian languages.
2. **NLLB-200 (1.3B / 3.3B):** Meta's highly reliable translation backbone covering low-resource and regional dialects.
3. **MADLAD-400 (3B / 7.2B):** Google's robust translation model with highly competitive Indic-to-English performance.
4. **Sarvam-Translate (4B):** Built by Sarvam AI in collaboration with AI4Bharat (on top of Gemma3-4B-IT). Designed specifically for document-level, contextually-aware translation across the 22 official Indian languages, addressing long-context translation needs that isolate sentence translators struggle with.

### 4. OCR (Document Vision & Layout Extraction)
1. **Indic-OCR by Bodhan AI (~0.8B):** Utilizes a custom Sarvam tokenizer and layout detector optimized for printed and handwritten Indic text.
2. **Surya OCR (~650M):** Highly accurate model for complex real-world layout analysis, table recognition, and reading order detection.
3. **PaddleOCR (PP-OCRv4, <100M):** Ultra-lightweight edge framework offering dedicated recognition weights for complex Indic scripts.
4. **EasyOCR (<100M):** Provides robust out-of-the-box support for major Indic scripts with virtually zero parameter overhead.

---

## 3. High-Level Architecture Blueprint

```text
========================================================================================
                                  +-------------+
                                  |  USER CLI   |
                                  +------+------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      [ setup <hf_repo> ]                               [ run --model ... ]
               |                                                   |
               v                                                   v
   =========================                           =========================
   PHASE 1: AOT COMPILER                               PHASE 2: STREAMING VM
   =========================                           =========================
               |                                                   |
   +-----------v-----------+                           +-----------v-----------+
   |   Hardware Profiler   |                           |    Manifest Loader    |
   | (OS, CUDA/MPS/CPU,    |                           | Reads:                |
   |  Total & Usable RAM)  |                           | manifests/<alias>.json|
   +-----------+-----------+                           +-----------+-----------+
               |                                                   |
   +-----------v-----------+                           +-----------v-----------+
   |    Model Profiler     |                           |   Engine Lifecycle    |
   | (HF API: Architecture,|                           |        Manager        |
   |  Params, Config)      |                           +-----------+-----------+
   +-----------+-----------+                                       |
               |                                       +-----------+-----------+
   +-----------v-----------+                           |                       |
   |    Decision Matrix    |                     [ Daemon Mode ]       [ In-Process ]
   | (Task + HW + Precision|                     (vLLM / llama-server) (CTranslate2/OCR)
   |  Engine Selection)    |                           |                       |
   +-----------+-----------+                     Spawn Subprocess      Direct Memory Load
               |                                 Healthcheck Loop              |
   +-----------v-----------+                           |                       |
   |   Artifact Compiler   |                           +-----------+-----------+
   | - Download weights    |                                       |
   | - Quantize / Convert  |                                       v
   | - Evict raw FP16      |                           +-----------------------+
   +-----------+-----------+                           |   Streaming Pipeline  |
               |                                       | - Chunked Reader      |
               v                                       | - Batch Dispatcher    |
   +-----------------------+                           +-----------+-----------+
   |    manifest.json      |                                       |
   |  (Execution Contract) |                           +-----------v-----------+
   +-----------------------+                           |     Task Adapter      |
                                                       | (Reason/Summ/Trn/OCR) |
                                                       +-----------+-----------+
                                                                   |
                                                       +-----------v-----------+
                                                       |     Output Sink       |
                                                       | - Stream to JSONL     |
                                                       | - On-the-fly Metrics  |
                                                       | - Auto Teardown       |
                                                       +-----------------------+

```

---

## 4. Key Architectural Decisions

### Decision 1: Hardware-Aware Precision Strategy (Deterministic Tiering)

To prevent out-of-memory (OOM) errors and maximize throughput, precision is selected dynamically based on task sensitivity and hardware capacity *before* any weights are downloaded.

* **OCR Pipeline:** Locked to **FP16 / BF16**. Low-bit quantization destroys the visual feature resolution required for complex Indic scripts (e.g., distinguishing *matras*).
* **Translation Pipeline:** Locked to **INT8** (via CTranslate2) or **GGUF Q4** (for decoder-based translators like Sarvam-Translate). Benchmarks prove INT8 maintains BLEU scores while halving memory requirements.
* **LLM Pipeline (Reasoning/Summ):**
* *Apple Silicon / CPU:* **GGUF Q4_K_M** to bypass severe memory-bandwidth bottlenecks.
* *NVIDIA GPU (Ample VRAM):* **FP16 / BF16** to utilize full PagedAttention and continuous batching via vLLM.
* *NVIDIA GPU (Constrained VRAM):* **AWQ / FP8** dynamically selected.



### Decision 2: Ahead-Of-Time (AOT) Weight Compilation

If the optimal artifact (e.g., GGUF or CT2-INT8) is not available on the Hugging Face Hub, the `setup` command compiles it locally.

* **Flow:** Download Base Safetensors $\rightarrow$ Boot Isolated Compiler $\rightarrow$ Convert to Target Format $\rightarrow$ **Delete Source FP16 Weights**.
* **Impact:** Saves ~50% disk space per model and ensures the `run` command boots instantly with zero compilation overhead.

### Decision 3: Zero-Bloat Isolated Runtimes

Mixing heavy CUDA dependencies (vLLM) with custom C++ bindings (CTranslate2) and Vision libraries (Torchvision) causes dependency conflicts.

* **Solution:** The framework acts as a runtime orchestrator.
* **llama.cpp:** Deployed as a pre-compiled, ~50MB standalone static binary.
* **vLLM / CTranslate2:** Installed into isolated, hidden micro-virtualenvs triggered dynamically.

### Decision 4: Asynchronous Streaming Data Engine

The `run` phase must never hold the entire dataset or output arrays in memory.

* Datasets (JSONL, Parquet, Image Directories) are read in sequential chunks.
* Outputs are appended directly to disk (`results.jsonl`) as soon as inference yields them.
* Metrics (BLEU, CER, ROUGE, tokens/sec) are calculated continuously on the fly.

### Decision 5: Hardcoded Framework State (Immutable Paths)

To eliminate path-resolution bugs and separate framework state from user data, all system directories are strictly hardcoded in an internal application configuration. The CLI accepts only model aliases and dataset paths.

```python
# indic_runner/config.py
import os
from pathlib import Path

# Single Source of Truth for all internal operations
BASE_DIR = Path(os.getenv("INDIC_RUNNER_HOME", Path.home() / ".indic-runner"))

DIRS = {
    "bin": BASE_DIR / "bin",             # Static binaries (e.g., llama-server)
    "envs": BASE_DIR / "envs",           # Isolated Python virtualenvs
    "models": BASE_DIR / "models",       # Compiled inference-ready weights
    "manifests": BASE_DIR / "manifests", # JSON execution contracts
}

```

---

## 5. The Execution Contract (`manifest.json`)

The `setup` command emits a declarative manifest. The `run` command reads this file and strictly executes it without requiring internet access or heuristic guessing.

**Example `manifests/sarvam-translate.json`:**

```json
{
  "manifest_version": "1.0",
  "model_alias": "sarvam-translate",
  "base_repo": "sarvamai/sarvam-translate",
  "task": "translation",
  "target_hardware": "mps",
  "precision": "q4_k_m",
  "engine": {
    "name": "llama.cpp",
    "mode": "daemon",
    "binary_or_env": ".indic-runner/bin/llama-server"
  },
  "paths": {
    "artifacts_dir": ".indic-runner/models/sarvam-translate-gguf",
    "tokenizer_dir": ".indic-runner/models/sarvam-translate-gguf"
  },
  "runtime_parameters": {
    "max_batch_size": 32,
    "context_length": 8192
  }
}

```

```

```