#!/usr/bin/env bash
# Real end-to-end smoke test of every roster model: setup, then run 2 rows.
# Usage: scripts/smoke_all.sh [alias ...]   (default: all 16, cheapest first)
# Results: $SMOKE_DIR/summary.txt. Needs HF_TOKEN in backend/.env (Airavata, IndicTrans2,
# Bodhan are auto-gated: accept each licence on the Hub once). Downloads ~150GB in total.
set -uo pipefail
cd "$(dirname "$0")/.."
SMOKE_DIR=${SMOKE_DIR:-$PWD/../smoke}; mkdir -p "$SMOKE_DIR"
SUMMARY="$SMOKE_DIR/summary.txt"; : > "$SUMMARY"

uv run python - "$SMOKE_DIR" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); d.mkdir(exist_ok=True)
w = lambda n, rows: (d / n).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
w("reasoning.jsonl", [{"id": "1", "input": "भारत की राजधानी क्या है?", "language": "hindi", "reference": "नई दिल्ली", "judge_rubric": "x"},
                      {"id": "2", "input": "2+3*4 कितना होता है?", "language": "hindi", "reference": "14", "judge_rubric": "x"}])
w("summarization.jsonl", [{"id": str(i), "language": "hindi", "reference": "x",
    "input": "भारत दक्षिण एशिया में स्थित एक विशाल देश है। इसकी जनसंख्या विश्व में सबसे अधिक है और यहाँ अनेक भाषाएँ बोली जाती हैं। " * 3} for i in (1, 2)])
w("translation_en.jsonl", [{"id": str(i), "language": "english", "target_language": "hindi", "reference": "x",
    "input": t} for i, t in ((1, "The weather is nice today."), (2, "Where is the railway station?"))])
w("translation_indic_en.jsonl", [{"id": str(i), "language": "hindi", "target_language": "english", "reference": "x",
    "input": t} for i, t in ((1, "आज मौसम अच्छा है।"), (2, "रेलवे स्टेशन कहाँ है?"))])
try:
    from PIL import Image, ImageDraw, ImageFont
    import glob
    fonts = glob.glob("/usr/share/fonts/**/*Devanagari*.ttf", recursive=True) or glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    f = ImageFont.truetype(fonts[0], 48) if fonts else ImageFont.load_default(48)
    rows = []
    for i, txt in ((1, "नमस्ते दुनिया"), (2, "भारत एक देश है")):
        im = Image.new("RGB", (900, 160), "white"); ImageDraw.Draw(im).text((20, 40), txt, font=f, fill="black")
        p = d / f"ocr{i}.png"; im.save(p)
        rows.append({"id": str(i), "image_path": str(p), "language": "hindi", "reference": txt})
    w("ocr.jsonl", rows)
except ImportError:
    print("Pillow missing: OCR dataset skipped (uv add --dev pillow)")
PY

# alias:task:dataset
CASES=(
  "paddleocr-pp-ocrv4:ocr:ocr" "surya-ocr:ocr:ocr" "indic-ocr-bodhan:ocr:ocr"
  "sarvam-1:reasoning:reasoning" "phi-4-mini-instruct:reasoning:reasoning"
  "qwen3-4b-instruct-2507:summarization:summarization"
  "indictrans2-en-indic:translation:translation_en" "indictrans2-indic-en:translation:translation_indic_en"
  "nllb-200:translation:translation_en" "sarvam-translate:translation:translation_en"
  "qwen2.5-7b-instruct:reasoning:reasoning" "qwen3-8b:reasoning:reasoning"
  "airavata-7b:summarization:summarization" "navarasa-2.0-7b:summarization:summarization"
  "madlad-400:translation:translation_en" "chandra-ocr:ocr:ocr"
)
WANT=("$@")
for c in "${CASES[@]}"; do
  IFS=: read -r alias task ds <<<"$c"
  if [ ${#WANT[@]} -gt 0 ] && [[ ! " ${WANT[*]} " =~ " $alias " ]]; then continue; fi
  echo "=== $alias ($task)"; t0=$SECONDS
  extra=(); [ "$task" = ocr ] && extra=(--ocr-variant normal)
  if ! uv run indic-runner setup "$alias" --task "$task" > "$SMOKE_DIR/$alias.setup.log" 2>&1; then
    echo "$alias  SETUP-FAIL  $(tail -1 "$SMOKE_DIR/$alias.setup.log")" | tee -a "$SUMMARY"; continue
  fi
  uv run indic-runner run --model "$alias" --dataset "$SMOKE_DIR/$ds.jsonl" "${extra[@]}" > "$SMOKE_DIR/$alias.run.log" 2>&1
  uv run python - "$alias" "$((SECONDS - t0))" "$SMOKE_DIR/$alias.run.log" >> "$SUMMARY" <<'PY'
import json, sys, re
alias, secs, log = sys.argv[1:4]
m = re.search(r"results written to (\S+)", open(log).read())
if not m: print(f"{alias}  RUN-FAIL  {open(log).read().strip().splitlines()[-1][:160]}"); sys.exit()
rows = json.load(open(m.group(1)))["results"]
ok = sum(r["status"] == "ok" for r in rows)
print(f"{alias}  {ok}/{len(rows)} ok  {secs}s total  sample: {(rows[0]['response'] or rows[0]['error'] or '')[:100]!r}")
PY
  tail -1 "$SUMMARY"
done
echo; cat "$SUMMARY"
