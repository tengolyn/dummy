"""OCR worker. One process per run, one branch per runtime (isolated env each).

Request: {"id", "image_path", "language", "prompt"?, "max_new_tokens"}.
Reply:   {"id", "text"} or {"id", "error"}.

  transformers -- Bodhan's IndicOCR (repo ships indic_ocr.py) or Chandra 2.
  surya        -- surya-ocr 0.6.x: detection + recognition (vikp/surya_rec2).
  paddleocr    -- PaddleOCR 3.x; weights come from the package's own per-language
                  models, since the roster repo (PP-OCRv4_server_rec) is en/zh only.
"""
import argparse, json, os, sys

# ISO-ish names used by the dataset -> codes each runtime wants.
SURYA_LANG = {
    "english": "en", "hindi": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te",
    "marathi": "mr", "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "punjabi": "pa",
    "odia": "or", "assamese": "as", "urdu": "ur", "nepali": "ne", "sanskrit": "sa",
}
# PaddleOCR ships recognisers only for some Indic scripts.
PADDLE_LANG = {
    "english": "en", "hindi": "hi", "marathi": "mr", "nepali": "ne", "sanskrit": "sa",
    "tamil": "ta", "telugu": "te", "kannada": "ka", "urdu": "ur",
}


def code(table, name, runtime):
    key = str(name).strip().lower()
    if key in table:
        return table[key]
    if key in table.values():
        return key
    raise ValueError(f"{runtime} has no recogniser for language {name!r}")


def load_hf(a):
    import torch
    dev = {"cuda": "cuda", "mps": "mps"}.get(a.device, "cpu")
    dtype = torch.bfloat16 if a.precision == "bf16" else torch.float16
    if os.path.isfile(os.path.join(a.artifacts, "indic_ocr.py")):  # Bodhan IndicOCR
        sys.path.insert(0, a.artifacts)
        from indic_ocr import IndicOCR
        parser = IndicOCR.from_pretrained(a.artifacts, device=dev)

        def run(req):
            return parser.parse(req["image_path"])["markdown"]
        return run
    # Chandra 2
    import transformers
    from PIL import Image
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        a.artifacts, dtype=dtype).to(dev).eval()
    model.processor = transformers.AutoProcessor.from_pretrained(a.artifacts)
    model.processor.tokenizer.padding_side = "left"

    def run(req):
        item = BatchInputItem(image=Image.open(req["image_path"]).convert("RGB"),
                              prompt_type="ocr_layout")
        raw = generate_hf([item], model, max_output_tokens=req["max_new_tokens"])[0].raw
        return parse_markdown(raw)
    return run


def load_surya(a):
    from PIL import Image
    from surya.model.detection.model import load_model as load_det, load_processor as load_det_p
    from surya.model.recognition.model import load_model as load_rec
    from surya.model.recognition.processor import load_processor as load_rec_p
    from surya.ocr import run_ocr
    det, det_p, rec, rec_p = load_det(), load_det_p(), load_rec(), load_rec_p()

    def run(req):
        lang = code(SURYA_LANG, req["language"], "surya")
        img = Image.open(req["image_path"]).convert("RGB")
        res = run_ocr([img], [[lang]], det, det_p, rec, rec_p)[0]
        return "\n".join(line.text for line in res.text_lines)
    return run


def load_paddle(a):
    engines = {}  # lang -> PaddleOCR, or the exception its constructor raised

    def build(lang):
        from paddleocr import PaddleOCR
        opts = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
        try:  # the roster model is PP-OCRv4; not every language ships a v4 recogniser
            return PaddleOCR(lang=lang, ocr_version="PP-OCRv4", **opts)
        except Exception:  # noqa: BLE001
            return PaddleOCR(lang=lang, **opts)

    def run(req):
        lang = code(PADDLE_LANG, req["language"], "paddleocr")
        if lang not in engines:
            try:
                engines[lang] = build(lang)
            except Exception as exc:  # noqa: BLE001
                # PaddleX cannot be re-initialised after a failed construction, so a
                # retry would only report "PDX has already been initialized" and hide this.
                engines[lang] = exc
        if isinstance(engines[lang], Exception):
            raise RuntimeError(f"PaddleOCR init failed for {lang!r}: {engines[lang]}")
        out = engines[lang].predict(req["image_path"])
        return "\n".join(t for page in out for t in page["rec_texts"])
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts"); ap.add_argument("--tokenizer")
    ap.add_argument("--device"); ap.add_argument("--precision")
    ap.add_argument("--runtime"); ap.add_argument("--alias")
    a = ap.parse_args()
    run = {"transformers": load_hf, "surya": load_surya, "paddleocr": load_paddle}[a.runtime](a)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            print(json.dumps({"id": req["id"], "text": run(req)}, ensure_ascii=False), flush=True)
        except Exception as exc:  # per-row failure must not kill the worker
            print(json.dumps({"id": req["id"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)


main()
