import hashlib
from pathlib import Path

from indic_runner.runtime.engines.base import Request


class OcrAdapter:
    def __init__(self, model_alias: str = ""):
        self.alias = model_alias

    def build(self, row: dict) -> Request:
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        from indic_runner.models.prompts import render_user, spec_for

        payload = {"image_path": str(path), "language": row["language"]}
        spec = spec_for(self.alias)
        if "ocr" in spec.user:  # VLM-style models take an instruction; det/rec models do not
            payload["prompt"] = render_user(spec, "ocr", row)
        return Request(row["id"], payload)

    def parse(self, text: str) -> tuple[str, str]:
        return text.strip(), ""

    def image_sha256(self, row: dict) -> str | None:
        h = hashlib.sha256()
        with open(row["image_path"], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
