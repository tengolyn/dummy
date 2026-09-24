"""Fixtures for run-phase tests: fake engines, fake llama-server, manifests."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

from indic_runner.runtime.engines.base import Engine, Response

FAKE_SERVER = f'''import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][-1]["content"]
        if "FAIL" in text:
            self.send_response(500); self.end_headers(); return
        out = json.dumps({{"choices": [{{"message": {{"content": "<think>t</think>echo:" + text}}}}]}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAKE_WORKER = '''
import json, sys
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if "FAIL" in r["text"]:
        print(json.dumps({"id": r["id"], "error": "boom"}), flush=True)
    else:
        print(json.dumps({"id": r["id"], "text": r["text"].upper()}), flush=True)
'''


class FakeEngine(Engine):
    """Deterministic engine; fails ids listed in `fail` (always), or `flaky` once."""

    def __init__(self, fail=(), flaky=()):
        self.fail, self.flaky = set(fail), set(flaky)
        self.started = self.stopped = False
        self.calls: list[list[str]] = []

    def start(self): self.started = True
    def stop(self): self.stopped = True

    def infer(self, batch, gen):
        self.calls.append([r.id for r in batch])
        out = []
        for r in batch:
            if r.id in self.fail:
                out.append(Response(r.id, error="always fails"))
            elif r.id in self.flaky:
                self.flaky.discard(r.id)
                out.append(Response(r.id, error="transient"))
            else:
                out.append(Response(r.id, f"<think>why</think>ans-{r.id}", latency_s=0.01))
        return out


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def reasoning_rows(n: int, prefix: str = "r") -> list[dict]:
    return [{"id": f"{prefix}{i}", "input": f"q{i}", "language": "hindi",
             "reference": "x", "judge_rubric": "y"} for i in range(n)]


def make_manifest_dict(tmp_path: Path, task="reasoning", engine="llama.cpp", mode="daemon",
                       binary=None, batch=2) -> dict:
    art = tmp_path / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "m.gguf").write_bytes(b"x")
    if binary is None and mode == "daemon":
        binary = str(install_fake_server(tmp_path))
    return {
        "manifest_version": "1.0", "model_alias": "fake", "base_repo": "o/fake", "revision": "abc",
        "task": task, "target_hardware": "cpu", "precision": "q4_k_m",
        "engine": {"name": engine, "mode": mode, "binary_or_env": binary or engine, "library_dir": None},
        "artifact_source": {"repo": "o/fake", "provenance": "converted"},
        "paths": {"artifacts_dir": str(art), "tokenizer_dir": str(art)},
        "runtime_parameters": {"max_batch_size": batch, "context_length": 512, "threads": 2},
    }


def make_executable(path: Path, python_source: str) -> Path:
    """A launchable stub. A /bin/sh wrapper avoids a shebang, which breaks when the
    interpreter path (the venv) contains a space."""
    body = path.with_suffix(".py")
    body.write_text(python_source)
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body}" "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def install_fake_server(tmp_path: Path) -> Path:
    return make_executable(tmp_path / "llama-server", FAKE_SERVER)
