from indic_runner.runtime.engines.base import Request
from indic_runner.runtime.task_adapters import build_llm_request, split_reasoning

# FLORES-200 style codes, used by NLLB and IndicTrans2.
FLORES = {
    "english": "eng_Latn", "hindi": "hin_Deva", "bengali": "ben_Beng", "tamil": "tam_Taml",
    "telugu": "tel_Telu", "marathi": "mar_Deva", "gujarati": "guj_Gujr", "kannada": "kan_Knda",
    "malayalam": "mal_Mlym", "punjabi": "pan_Guru", "odia": "ory_Orya", "assamese": "asm_Beng",
    "urdu": "urd_Arab", "sanskrit": "san_Deva", "nepali": "npi_Deva",
}


# MADLAD-400 uses ISO 639-1 style tags, written as a "<2xx>" source prefix.
MADLAD = {
    "english": "en", "hindi": "hi", "bengali": "bn", "tamil": "ta", "telugu": "te",
    "marathi": "mr", "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "punjabi": "pa",
    "odia": "or", "assamese": "as", "urdu": "ur", "nepali": "ne",
}


class UnknownLanguage(ValueError):
    pass


def flores(name: str) -> str:
    key = str(name).strip().lower()
    if key in FLORES:
        return FLORES[key]
    if key in FLORES.values() or "_" in name:
        return str(name)
    raise UnknownLanguage(f"no language code for {name!r}")


def madlad(name: str) -> str:
    key = str(name).strip().lower()
    if key in MADLAD:
        return MADLAD[key]
    if key in MADLAD.values():
        return key
    raise UnknownLanguage(f"MADLAD-400 has no language tag for {name!r}")


class TranslationAdapter:
    """Seq2seq engines get language codes; decoder (llama.cpp/vLLM) get a prompt."""

    def __init__(self, engine: str, model_alias: str = ""):
        from indic_runner.models.prompts import spec_for

        self.alias = model_alias
        self.spec = spec_for(model_alias)
        self.seq2seq = engine in ("ctranslate2", "transformers")

    def build(self, row: dict) -> Request:
        if not self.seq2seq:
            return build_llm_request("translation", self.alias, row)
        text = str(row["input"])
        if self.spec.codes == "madlad":
            # No source code and no target prefix: the target rides in the text.
            prefix = self.spec.src_prefix.replace("{tgt}", madlad(row["target_language"]))
            return Request(row["id"], {"text": prefix + text})
        return Request(row["id"], {
            "text": text,
            "src_code": flores(row["language"]),
            "tgt_code": flores(row["target_language"]),
        })

    def parse(self, text: str) -> tuple[str, str]:
        return split_reasoning(text) if not self.seq2seq else (text.strip(), "")

    def image_sha256(self, row):
        return None
