#!/usr/bin/env python
"""
Updated extractor that is compatible with:
 • spaCy ≤ 3.6 (thinc 8.1.x → NumPy < 2) – works with your current torch build
 • negspacy 0.1.9  (latest 0.1.x on PyPI)
No other pipeline files need to change.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import re, spacy

from symptom_net.constants import CANON_KEYS

# ───────────────────────────────────────────────────────────────────
# 0) Environment assumptions & quick sanity check
#    (You already installed these, but for reference)
#    pip install "spacy==3.6.*" "thinc==8.1.*" "numpy<2" "negspacy==0.1.9"
# ───────────────────────────────────────────────────────────────────

_SPACY_PATH = Path("models/extractor_ner/model")
if not _SPACY_PATH.exists():
    raise RuntimeError("spaCy NER model not found – run train_ner.py first.")

NER = spacy.load(str(_SPACY_PATH))

# ───────────────────────────────────────────────────────────────────
# 1) Negation detection – negspacy 0.1.x API
#    We **do not** instantiate Negex manually; instead we add the factory
#    by name so spaCy owns the component (required since spaCy 3.4).
# ───────────────────────────────────────────────────────────────────
from negspacy.termsets import termset               # present in 0.1.x
from negspacy.negation import Negex

_ts = termset("en_clinical")  # returns dict already in 0.1.x

if "negex" not in NER.pipe_names:
    NER.add_pipe(
        "negex",
        last=True,
        config={
            "neg_termset": _ts,
            "ent_types": ["ALL"],
            "chunk_prefix": ["no", "not", "without", "denies"],
        },
    )

# ───────────────────────────────────────────────────────────────────
# 2) Regex helpers (age, sex, duration)
# ───────────────────────────────────────────────────────────────────
_WORD2NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,
             "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
AGE_RGX = re.compile(r"\b(?P<num>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)[-\s]*(?:yrs?|years?)[-\s]*(?:old)?\b", re.I)
SEX_RGX = re.compile(r"\b(male|female|man|woman)\b", re.I)
DUR_RGX = re.compile(r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?P<unit>day|days|week|weeks|month|months)", re.I)
NEGATABLE = {"hemoptysis", "chest_tightness"}

def _w2n(w:str)->int: return int(_WORD2NUM.get(w.lower(), w))

def _age(t:str)->dict:
    m=AGE_RGX.search(t); return {"age": _w2n(m.group("num"))} if m else {}

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
            if k in res:
                if isinstance(res[k], list): res[k].append(ent.text)
                else: res[k]=[res[k], ent.text]
            else:
                res[k]=ent.text
    return res

# ───────────────────────────────────────────────────────────────────
# 3) Public function
# ───────────────────────────────────────────────────────────────────

def extract(text:str, prev:dict|None=None)->dict:
    prev=prev or {}
    out={**prev}
    for k,v in _ner(text).items(): out.setdefault(k,v)
    for fn in (_age,_sex,_dur):
        for k,v in fn(text).items(): out.setdefault(k,v)
    return out

# CLI quick check --------------------------------------------------
if __name__=="__main__":
    import sys, pprint
    pprint.pp(extract(" ".join(sys.argv[1:]) or input("> ")))
