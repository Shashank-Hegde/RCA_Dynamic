from __future__ import annotations
import re, json, spacy
from pathlib import Path
from typing import Dict, Any

from symptom_net.constants import CANON_KEYS

# ------------------------------------------------------------------
# ── Load spaCy pipeline + negation detector once at import time ───
# ------------------------------------------------------------------
_SPACY_PATH = Path("models/extractor_ner/model")
if not _SPACY_PATH.exists():
    raise RuntimeError("spaCy NER model not found → run train_ner.py first")

NER = spacy.load(str(_SPACY_PATH))
from negspacy.negation import Negex

# Load the proper negation term set correctly from get_termsets
from negspacy.termsets import get_termset
ts = get_termset("en_clinical")

NEG = Negex(
    nlp=NER,
    name="negex",
    neg_termset=ts,
    ent_types=["ALL"],
    extension_name="negex",
    chunk_prefix=["no", "not", "without", "denies"]
)

NER.add_pipe(NEG, last=True)

# ------------------------------------------------------------------
# ── Quick regex patterns (cheap & language─agnostic helpers) ─────
# ------------------------------------------------------------------
AGE_RGX = re.compile(
    r"\b(?P<num>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*[- ]?(year|yrs)[- ]?(old)?\b", re.I)
NUM_WORD = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

DURATION_RGX = re.compile(
    r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?P<unit>day|days|week|weeks|month|months)", re.I)

SEX_RGX = re.compile(r"\b(male|female|man|woman)\b", re.I)

NEGATABLE_SYMPTOM_KEYS = {"hemoptysis", "chest_tightness"}


# ------------------------------------------------------------------
# ── Helper functions (one per variable group) ──────────────────────────
# ------------------------------------------------------------------
def _num_word_to_int(w: str) -> int:
    return int(NUM_WORD.get(w.lower(), w))


def extract_age(text: str) -> dict:
    m = AGE_RGX.search(text)
    if not m:
        return {}
    n = m.group("num")
    age = _num_word_to_int(n)
    try:
        age = int(age)
    except ValueError:
        return {}
    return {"age": age}


def extract_sex(text: str) -> dict:
    m = SEX_RGX.search(text)
    if m:
        sex = "male" if m.group(1).lower() in ("male", "man") else "female"
        return {"sex": sex}
    return {}


def extract_durations(text: str) -> dict:
    durations = {}
    for num, unit in DURATION_RGX.findall(text):
        days = _num_word_to_int(num)
        days *= 30 if "month" in unit else 7 if "week" in unit else 1
        durations["unspecified"] = max(days, durations.get("unspecified", 0))
    return {"symptom_duration_map": durations} if durations else {}


def extract_spacy_ents(text: str) -> dict:
    doc = NER(text)
    result: Dict[str, Any] = {}
    for ent in doc.ents:
        key = ent.label_.lower()
        if key not in CANON_KEYS:
            continue
        if key in NEGATABLE_SYMPTOM_KEYS:
            result[key] = not ent._.negex
        else:
            if isinstance(result.get(key), list):
                result[key].append(ent.text)
            elif key in result:
                result[key] = [result[key], ent.text]
            else:
                result[key] = ent.text
    return result


# ------------------------------------------------------------------
# ── Main entry point used by the rest of the pipeline ──────────────────────────
# ------------------------------------------------------------------
def extract(text: str, prev: dict | None = None) -> dict:
    prev = prev or {}
    out: dict[str, Any] = {**prev}

    out.update({k: v for k, v in extract_spacy_ents(text).items()
                if k not in out})

    for fn in (extract_age, extract_sex, extract_durations):
        for k, v in fn(text).items():
            if k not in out:
                out[k] = v

    return out


# ------------------------------------------------------------------
# ── Manual CLI for quick testing ────────────────────────────────
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, pprint
    txt = sys.argv[1] if len(sys.argv) > 1 else input("Enter text: ")
    pprint.pp(extract(txt))
