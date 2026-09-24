"""transformers seq2seq worker (models with no CTranslate2 path, e.g. IndicTrans2)."""
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts"); ap.add_argument("--tokenizer")
    ap.add_argument("--device"); ap.add_argument("--precision")
    a = ap.parse_args()
    import torch, transformers
    dtype = torch.bfloat16 if a.precision == "bf16" else torch.float16
    dev = {"cuda": "cuda", "mps": "mps"}.get(a.device, "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        a.artifacts, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
    # IndicTrans2 needs its own pre/post-processing (normalisation, script
    # handling, language tags); the toolkit is the reference implementation.
    processor = None
    try:
        from IndicTransToolkit.processor import IndicProcessor
        processor = IndicProcessor(inference=True)
    except ImportError:
        pass
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            text = req["text"]
            indic = not hasattr(tok, "src_lang") and req.get("src_code") and req.get("tgt_code")
            if indic:
                if processor is None:
                    raise RuntimeError("IndicTransToolkit is not installed in this env")
                text = processor.preprocess_batch([text], src_lang=req["src_code"],
                                                  tgt_lang=req["tgt_code"])[0]
                enc = tok([text], src=True, return_tensors="pt", truncation=True).to(dev)
            else:
                if hasattr(tok, "src_lang") and req.get("src_code"):
                    tok.src_lang = req["src_code"]
                enc = tok(text, return_tensors="pt").to(dev)
            kw = {"max_new_tokens": req["max_new_tokens"], "num_beams": req.get("num_beams", 1),
                  "use_cache": True}
            if hasattr(tok, "src_lang") and req.get("tgt_code"):
                kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(req["tgt_code"])
            with torch.no_grad():
                out = model.generate(**enc, **kw)
            if indic:
                text = tok.batch_decode(out, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
                text = processor.postprocess_batch(text, lang=req["tgt_code"])[0]
            else:
                text = tok.decode(out[0], skip_special_tokens=True)
            print(json.dumps({"id": req["id"], "text": text},
                             ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"id": req["id"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)

main()
