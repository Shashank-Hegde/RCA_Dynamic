#!/usr/bin/env python
"""
Light‑weight extractor used by demo_predict_paths.py
----------------------------------------------------
 • Loads your trained spaCy NER from models/extractor_ner/
 • Adds a negation detector (negspacy) in a version‑agnostic way
 • Provides extract(text, prev={}) -> Dict[str, Any]
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import re, spacy

from symptom_net.constants import CANON_KEYS      # <-- your canonical key list

# ───────────────────────────────────────────────────────────────────
# 1) Load spaCy pipeline that you just trained
# ───────────────────────────────────────────────────────────────────
_SPACY_PATH = Path("models/extractor_ner/model")
if not _SPACY_PATH.exists():
    raise RuntimeError(
        "spaCy NER model not found.  Run scripts/train_ner.py first."
    )
NER = spacy.load(str(_SPACY_PATH))

# ───────────────────────────────────────────────────────────────────
# 2) Robust negspacy initialisation for ANY released version
# ───────────────────────────────────────────────────────────────────
from negspacy.negation import Negex  # API unchanged across versions

# fetch the English clinical term‑set; API changed between 0.1 and 0.4
try:
    # ≥0.4.0
    from negspacy.termsets import get_termset
    _ts = get_termset("en_clinical")
except ImportError:
    # 0.1.x – 0.3.x
    from negspacy.termsets import termset
    _ts = termset("en_clinical")

# make sure we end up with a plain dict (Negex expects this)
if not isinstance(_ts, dict):
    _ts = _ts.get_patterns()

NEG = Negex(
    nlp            = NER,
    name           = "negex",
    neg_termset    = _ts,
    ent_types      = ["ALL"],
    extension_name = "negex",
    chunk_prefix   = ["no", "not", "without", "denies"],
)
# add as the LAST component so we keep entities but enrich them with ._.negex
if "negex" not in NER.pipe_names:
    NER.add_pipe(NEG, last=True)

# ───────────────────────────────────────────────────────────────────
# 3) cheap regex helpers (age / sex / durations)
# ───────────────────────────────────────────────────────────────────
_WORD2NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,
             "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
AGE_RGX      = re.compile(r"\b(?P<num>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)[-\s]*(?:yrs?|years?)[-\s]*(?:old)?\b", re.I)
SEX_RGX      = re.compile(r"\b(male|female|man|woman)\b", re.I)
DUR_RGX      = re.compile(r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?P<unit>day|days|week|weeks|month|months)", re.I)
NEGATABLE    = {"hemoptysis", "chest_tightness"}

def _w2n(w:str) -> int: return int(_WORD2NUM.get(w.lower(), w))

# helper extractors ------------------------------------------------
def _age(t:str)->dict:
    m=AGE_RGX.search(t); 
    return {"age": _w2n(m.group("num"))} if m else {}

def _sex(t:str)->dict:
    m=SEX_RGX.search(t)
    if m: return {"sex": "male" if m.group(1).lower() in ("male","man") else "female"}
    return {}

def _dur(t:str)->dict:
    out={}
    for n,u in DUR_RGX.findall(t):
        days=_w2n(n); days*=30 if "month" in u else 7 if "week" in u else 1
        out["unspecified"]=max(days,out.get("unspecified",0))
    return {"symptom_duration_map":out} if out else {}

def _ner(t:str)->dict:
    doc=NER(t)
    res:Dict[str,Any]={}
    for ent in doc.ents:
        k=ent.label_.lower()
        if k not in CANON_KEYS: continue
        if k in NEGATABLE:
            res[k]=not ent._.negex
        else:
            # accumulate duplicates into list
            if k in res:
                if isinstance(res[k], list): res[k].append(ent.text)
                else: res[k]=[res[k],ent.text]
            else:
                res[k]=ent.text
    return res

# ───────────────────────────────────────────────────────────────────
# 4) public function
# ───────────────────────────────────────────────────────────────────
def extract(text:str, prev:dict|None=None)->dict:
    prev = prev or {}
    out  = {**prev}

    # spaCy NER + negation
    for k,v in _ner(text).items():
        out.setdefault(k,v)

    # regex fall‑backs
    for fn in (_age,_sex,_dur):
        for k,v in fn(text).items():
            out.setdefault(k,v)

    return out

# simple CLI -------------------------------------------------------
if __name__ == "__main__":
    import sys, pprint
    pprint.pp(extract(" ".join(sys.argv[1:]) or input("> ")))
