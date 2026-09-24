"""CTranslate2 seq2seq worker. Runs inside the isolated ctranslate2 env."""
import argparse, json, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts"); ap.add_argument("--tokenizer")
    ap.add_argument("--device"); ap.add_argument("--precision")
    a = ap.parse_args()
    import ctranslate2, transformers
    device = "cuda" if a.device == "cuda" else "cpu"
    tr = ctranslate2.Translator(a.artifacts, device=device)
    tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if req.get("src_code"):
                tok.src_lang = req["src_code"]
            toks = tok.convert_ids_to_tokens(tok.encode(req["text"]))
            prefix = [[req["tgt_code"]]] if req.get("tgt_code") else None
            res = tr.translate_batch([toks], target_prefix=prefix,
                                     beam_size=req.get("num_beams", 1),
                                     max_decoding_length=req["max_new_tokens"])
            out = res[0].hypotheses[0]
            if prefix and out and out[0] == req["tgt_code"]:
                out = out[1:]
            text = tok.decode(tok.convert_tokens_to_ids(out), skip_special_tokens=True)
            print(json.dumps({"id": req["id"], "text": text}, ensure_ascii=False), flush=True)
        except Exception as exc:  # per-row failure must not kill the worker
            print(json.dumps({"id": req["id"], "error": f"{type(exc).__name__}: {exc}"}), flush=True)

main()
