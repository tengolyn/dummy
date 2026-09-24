"""Per-model prompt specs: every roster alias renders a valid request for its task."""

from __future__ import annotations

import pytest

from indic_runner.models.prompts import PROMPTS, spec_for
from indic_runner.models.registry import MODEL_REGISTRY
from indic_runner.runtime.engines.openai_daemon import _text
from indic_runner.runtime.task_adapters import get_adapter
from indic_runner.runtime.task_adapters.translation import UnknownLanguage

ROW = {"id": "1", "language": "Hindi", "target_language": "Tamil", "input": "नमस्ते {x}",
       "reference": "r", "judge_rubric": "j"}
SEQ2SEQ = {"ctranslate2", "transformers"}


def _entries():
    return [(t, e["alias"]) for t, es in MODEL_REGISTRY.items() for e in es]


def test_roster_is_sixteen_distinct_models_all_with_specs():
    aliases = {a for _, a in _entries()}
    assert len(aliases) == 16
    assert aliases <= set(PROMPTS)


@pytest.mark.parametrize("task,alias", [(t, a) for t, a in _entries() if t != "ocr"])
def test_every_roster_pair_builds_a_request(task, alias):
    engine = "ctranslate2" if spec_for(alias).mode == "seq2seq" else "llama.cpp"
    req = get_adapter(task, engine, alias).build(ROW)
    assert req.payload
    assert "{input}" not in str(req.payload) and "{language}" not in str(req.payload)


def test_braces_in_dataset_text_survive():
    req = get_adapter("reasoning", "llama.cpp", "qwen3-8b").build(ROW)
    assert req.payload["messages"][0]["content"] == "नमस्ते {x}"


def test_base_and_tuned_models_use_completion_with_stops():
    sarvam = get_adapter("reasoning", "llama.cpp", "sarvam-1").build(ROW).payload
    assert sarvam["prompt"].endswith("Answer:") and sarvam["stop"]
    nav = get_adapter("summarization", "llama.cpp", "navarasa-2.0-7b").build(ROW).payload
    assert "### Instruction:" in nav["prompt"] and nav["prompt"].endswith("### Response:\n")
    air = get_adapter("summarization", "llama.cpp", "airavata-7b").build(ROW).payload
    assert air["prompt"].startswith("<|user|>") and air["prompt"].endswith("<|assistant|>\n")


def test_sarvam_translate_puts_target_in_system_prompt():
    msgs = get_adapter("translation", "llama.cpp", "sarvam-translate").build(ROW).payload["messages"]
    assert msgs[0] == {"role": "system", "content": "Translate the text below to Tamil."}
    assert msgs[1] == {"role": "user", "content": "नमस्ते {x}"}


def test_madlad_uses_target_prefix_and_no_codes():
    p = get_adapter("translation", "ctranslate2", "madlad-400").build(ROW).payload
    assert p == {"text": "<2ta> नमस्ते {x}"}
    with pytest.raises(UnknownLanguage):
        get_adapter("translation", "ctranslate2", "madlad-400").build({**ROW, "target_language": "Sanskrit"})


def test_nllb_and_indictrans_keep_flores_codes():
    for alias, eng in (("nllb-200", "ctranslate2"), ("indictrans2-en-indic", "transformers")):
        p = get_adapter("translation", eng, alias).build(ROW).payload
        assert (p["src_code"], p["tgt_code"]) == ("hin_Deva", "tam_Taml")


def test_reasoning_content_is_rewrapped_for_split():
    out = _text({"message": {"content": "42", "reasoning_content": "hmm"}})
    assert out == "<think>hmm</think>42"
    assert _text({"text": "plain"}) == "plain"
    assert get_adapter("reasoning", "llama.cpp", "qwen3-8b").parse(out) == ("42", "hmm")


# --- run-phase wiring: every roster model resolves to an env and a worker ---------------------

from indic_runner.setup import artifact_compiler as ac
from indic_runner.setup.decision_matrix import ExecutionPlan


def _plan(task, engine):
    return ExecutionPlan(task, engine, "in-process", "bf16", "safetensors")


@pytest.mark.parametrize("task,engine", [
    ("translation", "ctranslate2"), ("translation", "transformers"),
    ("ocr", "transformers"), ("ocr", "surya"), ("ocr", "paddleocr"),
])
def test_every_in_process_lane_has_an_env(task, engine):
    name, packages = ac.runtime_env(_plan(task, engine))
    assert name and packages


def test_conflicting_transformers_pins_live_in_separate_envs():
    hf, tr = ac.runtime_env(_plan("ocr", "transformers")), ac.runtime_env(_plan("translation", "transformers"))
    assert hf[0] != tr[0]
    assert ac.TRANSFORMERS_PIN in tr[1] and ac.TRANSFORMERS_PIN not in hf[1]
    assert ac.runtime_env(_plan("translation", "ctranslate2"))[0] == ac.CT2_ENV[0]


def test_daemons_have_no_env():
    assert ac.runtime_env(ExecutionPlan("reasoning", "llama.cpp", "daemon", "q4_k_m", "gguf")) is None


@pytest.mark.parametrize("runtime", ["transformers", "surya", "paddleocr"])
def test_ocr_runtimes_route_to_the_ocr_worker(tmp_path, monkeypatch, runtime):
    from runtime_helpers import make_manifest_dict
    from indic_runner.config import DIRS
    from indic_runner.runtime.engines.worker_engine import make_worker_engine
    from indic_runner.runtime.manifest_loader import Manifest
    from indic_runner.setup.env_manager import python_path

    monkeypatch.setitem(DIRS, "envs", tmp_path / "envs")
    python_path("ocr-x").parent.mkdir(parents=True)
    python_path("ocr-x").touch()
    raw = make_manifest_dict(tmp_path, task="ocr", engine=runtime, mode="in-process", binary="ocr-x")
    argv = make_worker_engine(Manifest(raw)).argv
    assert argv[1].endswith("ocr_worker.py")
    assert argv[argv.index("--runtime") + 1] == runtime
