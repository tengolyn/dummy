"""Run phase: loader, reader, adapters, sink, runner, engines, CLI."""

from __future__ import annotations

import json
import sys

import pytest
from runtime_helpers import (
    FAKE_WORKER, FakeEngine, install_fake_server, make_executable, make_manifest_dict,
    reasoning_rows, write_jsonl,
)

from indic_runner import cli
from indic_runner.runtime import manifest_loader as ml
from indic_runner.runtime import runner as rn
from indic_runner.runtime.engines.base import EngineError, GenerationConfig, Request
from indic_runner.runtime.engines.llama_cpp import LlamaCppEngine
from indic_runner.runtime.engines.worker_engine import WorkerEngine
from indic_runner.runtime.manifest_loader import Manifest, ManifestError, load_manifest
from indic_runner.runtime.output_sink import OutputSink, ResultValidationError
from indic_runner.runtime.streaming_pipeline import (
    DatasetError, batched, iter_rows, scan_dataset,
)
from indic_runner.runtime.task_adapters import get_adapter, split_reasoning
from indic_runner.runtime.task_adapters.translation import UnknownLanguage
from indic_runner.setup import manifest_writer as mw
from schemas.result_schema import RESULT_SCHEMA


@pytest.fixture
def home(tmp_path, monkeypatch):
    from indic_runner.config import DIRS
    for k in ("manifests", "runs", "envs"):
        monkeypatch.setitem(DIRS, k, tmp_path / "home" / k)
    return tmp_path


def install_manifest(home, **kw) -> Manifest:
    raw = make_manifest_dict(home, **kw)
    mw.write_manifest(raw)
    return load_manifest("fake")


# --- manifest loader -------------------------------------------------------

def test_loader_reads_valid_manifest(home):
    m = install_manifest(home)
    assert (m.alias, m.engine, m.max_batch_size) == ("fake", "llama.cpp", 2)


def test_loader_missing_manifest_says_run_setup(home):
    with pytest.raises(ManifestError, match="indic-runner setup"):
        load_manifest("nope")


def test_loader_rejects_corrupt_json(home):
    (home / "home" / "manifests").mkdir(parents=True)
    (home / "home" / "manifests" / "bad.json").write_text("{")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest("bad")


def test_loader_rejects_schema_violation_and_stale_version(home):
    raw = make_manifest_dict(home)
    (home / "home" / "manifests").mkdir(parents=True)
    path = home / "home" / "manifests" / "fake.json"
    path.write_text(json.dumps({**raw, "task": "chat"}))
    with pytest.raises(ManifestError, match="failed validation"):
        load_manifest("fake")
    path.write_text(json.dumps({**raw, "manifest_version": "0.1"}))
    with pytest.raises(ManifestError, match="manifest v0.1"):
        load_manifest("fake")


def test_loader_flags_missing_artifacts(home):
    raw = make_manifest_dict(home)
    raw["paths"]["artifacts_dir"] = str(home / "gone")
    mw.write_manifest(raw)
    with pytest.raises(ManifestError, match="missing files"):
        load_manifest("fake")


# --- dataset reader --------------------------------------------------------

def test_reader_streams_and_validates(tmp_path):
    p = write_jsonl(tmp_path / "d.jsonl", reasoning_rows(3))
    assert [r["id"] for r in iter_rows(p, "reasoning")] == ["r0", "r1", "r2"]
    info = scan_dataset(p, "reasoning")
    assert (info.row_count, info.languages) == (3, ["hindi"])


def test_reader_reports_line_of_missing_column(tmp_path):
    rows = reasoning_rows(2)
    del rows[1]["judge_rubric"]
    p = write_jsonl(tmp_path / "d.jsonl", rows)
    with pytest.raises(DatasetError, match=r"d.jsonl:2: missing reasoning columns \['judge_rubric'\]"):
        list(iter_rows(p, "reasoning"))


def test_reader_rejects_bad_json_duplicates_and_format(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("{oops\n")
    with pytest.raises(DatasetError, match="invalid JSON"):
        list(iter_rows(p, "reasoning"))
    rows = reasoning_rows(1) * 2
    with pytest.raises(DatasetError, match="duplicate id"):
        list(iter_rows(write_jsonl(p, rows), "reasoning"))
    with pytest.raises(DatasetError, match="unsupported dataset format"):
        (tmp_path / "d.csv").write_text("")
        list(iter_rows(tmp_path / "d.csv", "reasoning"))
    with pytest.raises(DatasetError, match="not found"):
        list(iter_rows(tmp_path / "zz.jsonl", "reasoning"))


def test_batched_chunks_and_rejects_zero():
    assert [len(b) for b in batched(iter(range(5)), 2)] == [2, 2, 1]
    with pytest.raises(ValueError):
        list(batched([], 0))


# --- adapters --------------------------------------------------------------

def test_split_reasoning_variants():
    assert split_reasoning("<think>a</think>b") == ("b", "a")
    assert split_reasoning("a</think>b") == ("b", "a")
    assert split_reasoning("plain") == ("plain", "")


def test_translation_adapter_seq2seq_vs_decoder():
    row = {"id": "1", "input": "hello", "language": "English", "target_language": "Hindi"}
    seq = get_adapter("translation", "ctranslate2").build(row).payload
    assert (seq["src_code"], seq["tgt_code"]) == ("eng_Latn", "hin_Deva")
    dec = get_adapter("translation", "llama.cpp").build(row).payload
    assert "Hindi" in dec["messages"][0]["content"]
    with pytest.raises(UnknownLanguage):
        get_adapter("translation", "ctranslate2").build({**row, "language": "Klingon"})


def test_ocr_adapter_hashes_image_and_rejects_missing(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"abc")
    a = get_adapter("ocr", "surya")
    row = {"id": "1", "image_path": str(img), "language": "hindi", "reference": ""}
    assert a.build(row).payload["image_path"] == str(img)
    assert a.image_sha256(row) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(FileNotFoundError):
        a.build({**row, "image_path": str(tmp_path / "no.png")})


# --- output sink -----------------------------------------------------------

def _row(i, status="ok"):
    return {"id": str(i), "input": "q", "response": "a", "reasoning": "", "latency_s": 0.1,
            "status": status, "error": None, "image_sha256": None}


def test_sink_rejects_schema_violations(tmp_path):
    s = OutputSink(tmp_path)
    bad = _row(1)
    del bad["response"]
    with pytest.raises(ResultValidationError):
        s.append(bad)


def test_sink_resume_drops_torn_line(tmp_path):
    s = OutputSink(tmp_path)
    s.append(_row(1))
    s.close()
    with s.partial.open("a") as fh:
        fh.write('{"id": "2", "inp')
    assert OutputSink(tmp_path, resume=True).done_ids() == {"1"}
    assert s.partial.read_text().count("\n") == 1


def test_sink_fresh_start_discards_old_partial(tmp_path):
    s = OutputSink(tmp_path)
    s.append(_row(1))
    s.close()
    assert OutputSink(tmp_path).done_ids() == set()


# --- runner (fake engine) --------------------------------------------------

def run_fake(home, engine, rows, **opt):
    m = install_manifest(home)
    ds = write_jsonl(home / "d.jsonl", rows)
    return rn.run(m, ds, rn.RunOptions(**opt), engine=engine)


def test_runner_end_to_end_output_matches_schema(home):
    eng = FakeEngine()
    res = run_fake(home, eng, reasoning_rows(5))
    doc = json.loads(res.results_file.read_text())
    from indic_runner.runtime.output_sink import validate
    validate(RESULT_SCHEMA, doc, "doc")
    assert res.status == "completed" and (res.successful, res.failed) == (5, 0)
    assert [r["id"] for r in doc["results"]] == [f"r{i}" for i in range(5)]
    assert doc["results"][0]["response"] == "ans-r0" and doc["results"][0]["reasoning"] == "why"
    assert doc["manifest"]["dataset"]["row_count"] == 5
    assert eng.calls == [["r0", "r1"], ["r2", "r3"], ["r4"]]  # batch size honoured
    assert eng.started and eng.stopped


def test_runner_retries_transient_and_records_permanent_failures(home):
    eng = FakeEngine(fail={"r1"}, flaky={"r2"})
    res = run_fake(home, eng, reasoning_rows(3), max_retries=1)
    rows = {r["id"]: r for r in json.loads(res.results_file.read_text())["results"]}
    assert rows["r2"]["status"] == "ok"
    assert rows["r1"]["status"] == "error" and rows["r1"]["error"] == "always fails"
    assert res.status == "partial" and res.failed == 1


def test_runner_all_failed_status(home):
    res = run_fake(home, FakeEngine(fail={"r0"}), reasoning_rows(1), max_retries=0)
    assert res.status == "failed"


def test_runner_bad_dataset_fails_before_engine_starts(home):
    eng = FakeEngine()
    rows = reasoning_rows(2)
    del rows[1]["input"]
    with pytest.raises(DatasetError):
        run_fake(home, eng, rows)
    assert not eng.started


def test_runner_stops_engine_on_error(home):
    class Boom(FakeEngine):
        def infer(self, batch, gen):
            raise EngineError("dead")
    eng = Boom()
    with pytest.raises(EngineError):
        run_fake(home, eng, reasoning_rows(2))
    assert eng.stopped


def test_runner_row_build_failure_is_row_level(home):
    m = install_manifest(home, task="ocr", engine="surya", mode="in-process")
    rows = [{"id": "a", "image_path": str(home / "missing.png"), "language": "hindi", "reference": ""}]
    ds = write_jsonl(home / "o.jsonl", rows)
    res = rn.run(m, ds, rn.RunOptions(ocr_variant="normal"), engine=FakeEngine())
    out = json.loads(res.results_file.read_text())
    assert out["results"][0]["status"] == "error" and "image not found" in out["results"][0]["error"]
    assert out["manifest"]["dataset"]["ocr_variant"] == "normal"


def test_ocr_variant_rules(home):
    m = install_manifest(home, task="ocr", engine="surya", mode="in-process")
    ds = write_jsonl(home / "o.jsonl", [])
    with pytest.raises(ValueError, match="ocr_variant"):
        rn.run(m, ds, engine=FakeEngine())
    m2 = install_manifest(home)
    with pytest.raises(ValueError, match="only applies"):
        rn.run(m2, ds, rn.RunOptions(ocr_variant="normal"), engine=FakeEngine())


def test_runner_resume_skips_done_rows_and_counts_whole_run(home):
    rows = reasoning_rows(4)
    first = run_fake(home, FakeEngine(), rows[:2])
    # Same dataset name is irrelevant; resume with the full dataset under the same run id.
    eng = FakeEngine()
    m = load_manifest("fake")
    ds = write_jsonl(home / "full.jsonl", rows)
    res = rn.run(m, ds, rn.RunOptions(resume_run_id=first.run_id), engine=eng)
    assert sum(len(c) for c in eng.calls) == 2  # only r2, r3 re-ran
    doc = json.loads(res.results_file.read_text())
    assert len(doc["results"]) == 4 and doc["manifest"]["run"]["successful_count"] == 4
    assert doc["manifest"]["run"]["resumed"] is True


# --- real engines against fakes --------------------------------------------

def test_llama_cpp_engine_against_fake_server(home):
    server = install_fake_server(home)
    m = install_manifest(home, binary=str(server))
    eng = LlamaCppEngine(m)
    eng.startup_timeout_s = 20
    eng.start()
    try:
        out = eng.infer([Request("a", {"messages": [{"role": "user", "content": "hi"}]}),
                         Request("b", {"messages": [{"role": "user", "content": "FAIL"}]})],
                        GenerationConfig(timeout_s=5))
    finally:
        eng.stop()
    assert out[0].text.endswith("echo:hi") and out[0].error is None
    assert out[1].error is not None  # per-row HTTP failure, engine survives
    assert eng._proc.poll() is not None  # daemon torn down


def test_llama_cpp_end_to_end_via_runner(home):
    server = install_fake_server(home)
    m = install_manifest(home, binary=str(server))
    ds = write_jsonl(home / "d.jsonl", reasoning_rows(3))
    eng = LlamaCppEngine(m)
    eng.startup_timeout_s = 20
    res = rn.run(m, ds, rn.RunOptions(), engine=eng)
    doc = json.loads(res.results_file.read_text())
    assert res.status == "completed"
    assert doc["results"][1]["response"] == "echo:q1" and doc["results"][1]["reasoning"] == "t"


def test_llama_cpp_startup_failure_surfaces_log(home, tmp_path):
    bad = tmp_path / "bad-server"
    make_executable(bad, "import sys; print('kaboom'); sys.exit(3)\n")
    m = install_manifest(home, binary=str(bad))
    eng = LlamaCppEngine(m)
    with pytest.raises(EngineError, match="kaboom"):
        eng.start()
    eng.stop()


def test_worker_engine_protocol(tmp_path):
    script = tmp_path / "w.py"
    script.write_text(FAKE_WORKER)
    eng = WorkerEngine([sys.executable, str(script)], startup_timeout_s=10)
    eng.start()
    try:
        out = eng.infer([Request("1", {"text": "abc"}), Request("2", {"text": "FAIL"})], GenerationConfig())
    finally:
        eng.stop()
    assert (out[0].text, out[1].error) == ("ABC", "boom")


def test_worker_engine_reports_bad_startup(tmp_path):
    script = tmp_path / "w.py"
    script.write_text("print('not json')\n")
    with pytest.raises(EngineError, match="failed to start"):
        WorkerEngine([sys.executable, str(script)], startup_timeout_s=5).start()


def test_unsupported_engine_is_a_clear_error(home):
    from indic_runner.runtime.engine_lifecycle import build_engine
    m = install_manifest(home, task="ocr", engine="easyocr", mode="in-process")
    with pytest.raises(EngineError, match="no run-phase worker"):
        build_engine(m)


# --- CLI -------------------------------------------------------------------

def test_cli_run_reports_missing_manifest(home, capsys, monkeypatch):
    ds = write_jsonl(home / "d.jsonl", reasoning_rows(1))
    monkeypatch.setattr(sys, "argv", ["indic-runner", "run", "--model", "zzz", "--dataset", str(ds)])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1 and "indic-runner setup zzz" in capsys.readouterr().err


def test_cli_run_end_to_end(home, capsys, monkeypatch):
    server = install_fake_server(home)
    install_manifest(home, binary=str(server))
    ds = write_jsonl(home / "d.jsonl", reasoning_rows(2))
    monkeypatch.setattr(sys, "argv", ["indic-runner", "run", "--model", "fake", "--dataset", str(ds)])
    monkeypatch.setattr(LlamaCppEngine, "startup_timeout_s", 20)
    with pytest.raises(SystemExit) as e:
        cli.main()
    out = capsys.readouterr().out
    assert e.value.code == 0 and "completed (2 ok, 0 failed)" in out


def test_llama_cpp_kv_cache_is_bounded(home):
    """Regression: 16 slots x 8192 ctx asked a real 2.5B model for ~14GB of KV cache."""
    raw = make_manifest_dict(home, batch=64)
    raw["runtime_parameters"]["context_length"] = 32768
    mw.write_manifest(raw)
    cmd = LlamaCppEngine(load_manifest("fake")).command()
    assert cmd[cmd.index("-c") + 1] == str(4096 * 8)
    assert cmd[cmd.index("--parallel") + 1] == "8"
