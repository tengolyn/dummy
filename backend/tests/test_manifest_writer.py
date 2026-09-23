"""Manifest assembly, validation and atomic write."""

from __future__ import annotations

import json

import pytest
from helpers import make_hw, make_model

from indic_runner.setup import manifest_writer as mw
from indic_runner.setup.decision_matrix import select_plan

VALID = {
    "manifest_version": "1.0",
    "model_alias": "sarvam-translate",
    "base_repo": "sarvamai/sarvam-translate",
    "task": "translation",
    "target_hardware": "cpu",
    "precision": "int8",
    "engine": {"name": "ctranslate2", "mode": "in-process", "binary_or_env": "/env"},
    "artifact_source": {"repo": "sarvamai/sarvam-translate", "provenance": "official"},
    "paths": {"artifacts_dir": "/models/x-ct2", "tokenizer_dir": "/models/x-tok"},
    "runtime_parameters": {"max_batch_size": 32, "context_length": 8192},
}


def test_valid_manifest_passes():
    mw.validate_manifest(VALID)


@pytest.mark.parametrize("field", sorted(VALID))
def test_every_required_top_level_field_is_enforced(field):
    if field == "revision":
        return
    incomplete = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(mw.ManifestValidationError, match=field):
        mw.validate_manifest(incomplete)


@pytest.mark.parametrize(
    "path,value",
    [
        ("task", "evaluation"),          # not a runner task
        ("target_hardware", "tpu"),
        ("precision", "q2_k"),
    ],
)
def test_enum_violations_are_rejected(path, value):
    broken = dict(VALID, **{path: value})
    with pytest.raises(mw.ManifestValidationError, match=path):
        mw.validate_manifest(broken)


def test_engine_enum_is_enforced():
    broken = dict(VALID, engine=dict(VALID["engine"], name="onnxruntime"))
    with pytest.raises(mw.ManifestValidationError, match="engine.name"):
        mw.validate_manifest(broken)


def test_nested_required_field_is_enforced():
    broken = dict(VALID, paths={"artifacts_dir": "/only"})
    with pytest.raises(mw.ManifestValidationError, match="tokenizer_dir"):
        mw.validate_manifest(broken)


def test_wrong_type_is_rejected():
    broken = dict(VALID, runtime_parameters={"max_batch_size": "32", "context_length": 8192})
    with pytest.raises(mw.ManifestValidationError, match="max_batch_size"):
        mw.validate_manifest(broken)


def test_boolean_is_not_accepted_as_integer():
    broken = dict(VALID, runtime_parameters={"max_batch_size": True, "context_length": 8192})
    with pytest.raises(mw.ManifestValidationError, match="max_batch_size"):
        mw.validate_manifest(broken)


def test_build_manifest_produces_absolute_paths(tmp_path):
    hw = make_hw("cpu")
    plan = select_plan("reasoning", hw, make_model(params_b=1.0))
    manifest = mw.build_manifest(
        model_alias="alias",
        base_repo="org/repo",
        revision="sha",
        plan=plan,
        hardware=hw,
        artifacts_dir=tmp_path / "artifacts",
        tokenizer_dir=tmp_path / "tok",
        binary_or_env="/bin/llama-server",
        library_dir=tmp_path / "lib",
    )
    assert manifest["paths"]["artifacts_dir"].startswith("/")
    assert manifest["engine"]["library_dir"] == str(tmp_path / "lib")
    assert manifest["target_hardware"] == "cpu"


def test_write_manifest_round_trips(tmp_path, monkeypatch):
    monkeypatch.setitem(mw.DIRS, "manifests", tmp_path / "manifests")
    path = mw.write_manifest(VALID)
    assert path.exists()
    assert json.loads(path.read_text()) == VALID


def test_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setitem(mw.DIRS, "manifests", tmp_path / "manifests")
    mw.write_manifest(VALID)
    assert list((tmp_path / "manifests").glob("*.tmp")) == []


def test_invalid_manifest_is_never_written(tmp_path, monkeypatch):
    monkeypatch.setitem(mw.DIRS, "manifests", tmp_path / "manifests")
    with pytest.raises(mw.ManifestValidationError):
        mw.write_manifest(dict(VALID, precision="bogus"))
    assert not (tmp_path / "manifests").exists() or not list(
        (tmp_path / "manifests").iterdir()
    )


@pytest.mark.parametrize(
    "engine", ["llama.cpp", "vllm", "ctranslate2", "transformers", "surya", "paddleocr", "easyocr"]
)
def test_every_engine_the_matrix_can_select_is_accepted(engine):
    mw.validate_manifest(dict(VALID, engine=dict(VALID["engine"], name=engine)))


def test_manifest_engine_enum_covers_the_decision_matrix():
    """The schema must not reject a plan the matrix is able to produce."""
    from indic_runner.setup.decision_matrix import OCR_ARTIFACT_FORMAT
    from schemas.manifest_schema import MANIFEST_SCHEMA

    allowed = set(MANIFEST_SCHEMA["properties"]["engine"]["properties"]["name"]["enum"])
    assert set(OCR_ARTIFACT_FORMAT) <= allowed
