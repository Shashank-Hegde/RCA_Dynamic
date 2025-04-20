#!/usr/bin/env python

import spacy
import json
import pathlib
import argparse
import subprocess
from spacy.tokens import DocBin

ALL_LABELS = [
    "AGE", "SEX", "ETHNICITY", "LOCATION", "REGION", "SOCIOECONOMIC_STATUS",
    "PAST_CONDITIONS", "SURGERIES", "HOSPITALISATIONS", "CHRONIC_ILLNESSES",
    "MEDICATION_HISTORY", "IMMUNISATION_STATUS", "ALLERGIES",
    "FAMILY_HISTORY", "DIET", "PHYSICAL_ACTIVITY", "SLEEP_PATTERN",
    "ALCOHOL", "TOBACCO", "MENTAL_HEALTH", "WORK_STRESS",
    "ENVIRONMENTAL_EXPOSURE", "HOUSING", "CLEAN_WATER", "OCCUPATION"
]

def jsonl_to_docbin(path, nlp):
    db = DocBin()
    for line in open(path):
        j = json.loads(line)
        doc = nlp.make_doc(j["text"])
        spans = []
        for k, v in j["extracted"].items():
            if v in (None, "", []):
                continue
            label = k.upper()
            if label not in ALL_LABELS:
                continue
            val = str(v)
            start = j["text"].lower().find(val.lower())
            if start == -1:
                continue
            end = start + len(val)
            span = doc.char_span(start, end, label=label)
            if span and not any(span.start < s.end and s.start < span.end for s in spans):
                spans.append(span)
        doc.ents = spans
        db.add(doc)
    return db

def make_config(output):
    cfg = """\
[paths]
train = null
dev = null

[system]
gpu_allocator = "pytorch"

[nlp]
lang = "en"
pipeline = ["transformer", "ner"]
batch_size = 128

[components]

[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "roberta-base"
tokenizer_config = {}
transformer_config = {}
mixed_precision = true

[components.ner]
factory = "ner"

[training]
max_epochs = 10
dropout = 0.1
seed = 42
gpu_allocator = "pytorch"

[training.optimizer]
@optimizers = "Adam"

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
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for lbl in ALL_LABELS:
        ner.add_label(lbl)

    print(f"ℹ Converting JSONL to .spacy format …")
    db_train = jsonl_to_docbin(train_jsonl, nlp)
    db_train.to_disk(out_dir / "train.spacy")
    db_val = jsonl_to_docbin(val_jsonl, nlp)
    db_val.to_disk(out_dir / "val.spacy")

    print(f"ℹ Writing config file to {out_dir/'config.cfg'} …")
    make_config(out_dir)

    print(f"ℹ Starting spaCy NER training on CPU")
    try:
        subprocess.run([
            "python", "-m", "spacy", "train",
            str(out_dir / "config.cfg"),
            "--output", str(out_dir),
            "--paths.train", str(out_dir / "train.spacy"),
            "--paths.dev", str(out_dir / "val.spacy"),
            "--gpu-id", "-1"  # Force CPU
        ], check=True)
    except subprocess.CalledProcessError as e:
        print("✘ spaCy training failed.")
        print(e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/synth/train.jsonl")
    p.add_argument("--val", default="data/synth/val.jsonl")
    p.add_argument("--out", default="models/extractor_ner")
    a = p.parse_args()
    main(a.train, a.val, a.out)
