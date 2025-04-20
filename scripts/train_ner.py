#!/usr/bin/env python
"""
Train a token‑level extractor that recognises every variable except the
two symptom maps (they are numeric; handled by rules).
"""

import spacy, argostranslate.package, json, pathlib, argparse, random
from spacy.tokens import DocBin

ALL_LABELS = [
 "AGE","SEX","ETHNICITY","LOCATION","REGION","SOCIOECONOMIC_STATUS",
 "PAST_CONDITIONS","SURGERIES","HOSPITALISATIONS","CHRONIC_ILLNESSES",
 "MEDICATION_HISTORY","IMMUNISATION_STATUS","ALLERGIES",
 "FAMILY_HISTORY","DIET","PHYSICAL_ACTIVITY","SLEEP_PATTERN",
 "ALCOHOL","TOBACCO","MENTAL_HEALTH","WORK_STRESS",
 "ENVIRONMENTAL_EXPOSURE","HOUSING","CLEAN_WATER","OCCUPATION"
]

def jsonl_to_docbin(path, nlp):
    db=DocBin()
    for line in open(path):
        j=json.loads(line)
        doc=nlp.make_doc(j["text"])
        spans=[]
        for k,v in j["extracted"].items():
            if v in (None,"",[]):
                continue
            label=k.upper()
            if label not in ALL_LABELS:             # skip maps
                continue
            # find *first* occurrence
            val=str(v)
            start=j["text"].lower().find(val.lower())
            if start==-1: continue
            end=start+len(val)
            span=doc.char_span(start,end,label=label)
            if span and not any(span.start < s.end and s.start < span.end for s in spans):
                spans.append(span)

        doc.ents=spans; db.add(doc)
    return db

def make_config(output):
    cfg = f"""
[paths]
train = null
dev = null

[system]
gpu_allocator = "pytorch"

[nlp]
lang = "en"
pipeline = ["tok2vec","ner"]
batch_size = 128

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2VecTransformer.v3"
name = "roberta-base"
tokenizer_config = {{}}
transformer_config = {{}}
pooling = {{}}
grad_scaler_config = {{}}
mixed_precision = false

[components.ner]
factory = "ner"

[training]
dev_corpus = "corpus.dev"
train_corpus = "corpus.train"
max_epochs = 10
dropout = 0.1
accumulate_gradient = 1
patience = 3000
eval_frequency = 200
frozen_components = []
seed = 42
gpu_allocator = "pytorch"
optimizer = {{"@optimizers": "Adam"}}

[training.optimizer.learn_rate]
@decay_rate = "compounding"
initial_rate = 0.00005
decay = 0.01
t = 1.0

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = "models/extractor_ner/train.spacy"

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = "models/extractor_ner/val.spacy"
"""
    (output / "config.cfg").write_text(cfg)

def main(train_jsonl, val_jsonl, out_dir):
    out_dir=pathlib.Path(out_dir); out_dir.mkdir(exist_ok=True,parents=True)
    nlp=spacy.blank("en"); ner=nlp.add_pipe("ner")
    for lbl in ALL_LABELS: ner.add_label(lbl)
    db_train=jsonl_to_docbin(train_jsonl,nlp); db_train.to_disk(out_dir/"train.spacy")
    db_val  =jsonl_to_docbin(val_jsonl  ,nlp); db_val.to_disk(out_dir/"val.spacy")
    make_config(out_dir)
    # call spaCy train CLI
    import subprocess, sys
    subprocess.run(["python","-m","spacy","train",str(out_dir/"config.cfg"),
                    "--output",str(out_dir),
                    "--paths.train",str(out_dir/"train.spacy"),
                    "--paths.dev",str(out_dir/"val.spacy"),
                    "--gpu-id","-1"],check=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",default="data/synth/train.jsonl")
    p.add_argument("--val",  default="data/synth/val.jsonl")
    p.add_argument("--out",  default="models/extractor_ner")
    a=p.parse_args(); main(a.train,a.val,a.out)
