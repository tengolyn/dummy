"""The run-output contract: manifest + results."""

from __future__ import annotations

from schemas.result_schema import RESULT_SCHEMA

MANIFEST = RESULT_SCHEMA["properties"]["manifest"]["properties"]
ITEM = RESULT_SCHEMA["properties"]["results"]["items"]


def test_top_level_keys_required():
    assert RESULT_SCHEMA["required"] == ["manifest", "results"]


def test_result_item_response_field_matches_required():
    assert "response" in ITEM["properties"]
    assert "response" in ITEM["required"]
    assert "prediction" not in ITEM["required"]


def test_every_required_result_field_is_declared():
    assert set(ITEM["required"]) <= set(ITEM["properties"])


def test_ocr_variant_is_on_the_dataset_manifest():
    variant = MANIFEST["dataset"]["properties"]["ocr_variant"]
    assert variant["type"] == ["string", "null"]
    assert set(variant["enum"]) == {"normal", "scanned", None}
    assert "ocr_variant" in MANIFEST["dataset"]["required"]


def test_nullable_fields_accept_null():
    for field in ("error", "image_sha256"):
        assert ITEM["properties"][field]["type"] == ["string", "null"]


def test_runtime_records_telemetry_but_no_quality_metrics():
    runtime = set(MANIFEST["runtime"]["properties"])
    assert {"latency_p50_s", "throughput_rows_per_s"} <= runtime
    assert not runtime & {"bleu", "cer", "rouge", "accuracy"}
