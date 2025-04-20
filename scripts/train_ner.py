#!/usr/bin/env python
"""
Train a lightweight spaCy NER extractor (no transformers) that recognises
all context variables.  Works on CPU; switch --gpu‑id 0 for GPU.
"""

import json, pathlib, argparse, subprocess, spacy
from spacy.tokens import DocBin


ALL_LABELS = [
    "AGE","SEX","ETHNICITY","LOCATION","REGION","SOCIOECONOMIC_STATUS",
    "PAST_CONDITIONS","SURGERIES","HOSPITALISATIONS","CHRONIC_ILLNESSES",
    "MEDICATION_HISTORY","IMMUNISATION_STATUS","ALLERGIES","FAMILY_HISTORY",
    "DIET","PHYSICAL_ACTIVITY","SLEEP_PATTERN","ALCOHOL","TOBACCO",
    "MENTAL_HEALTH","WORK_STRESS","ENVIRONMENTAL_EXPOSURE","HOUSING",
    "CLEAN_WATER","OCCUPATION"
]

# ---------------------------------------------------------------------------
# Helper: convert JSONL → spaCy DocBin
# ---------------------------------------------------------------------------
def jsonl_to_docbin(path: str, nlp) -> DocBin:
    db = DocBin()
    for line in open(path, "r", encoding="utf-8"):
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


# ---------------------------------------------------------------------------
# Helper: write a VALID spaCy‑3 config that passes validation
# ---------------------------------------------------------------------------
def make_config(output):
    cfg = """\
[paths]
train = "models/extractor_ner/train.spacy"
dev = "models/extractor_ner/val.spacy"

[system]
seed = 42

[nlp]
lang = "en"
pipeline = ["tok2vec", "ner"]
batch_size = 128

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@layers = "HashEmbed.v1"
nO = 96
nV = 20000

[components.tok2vec.model.encode]
@layers = "MaxoutWindowEncoder.v2"
width = 96
window_size = 1
maxout_pieces = 3
depth = 2

[components.ner]
factory = "ner"

[training]
train_corpus = "corpora.train"
dev_corpus = "corpora.dev"
seed = 42
dropout = 0.1
max_epochs = 10
patience = 5
gpu_allocator = "pytorch"

[training.optimizer]
@optimizers = "Adam.v1"
learn_rate = 0.0001

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = "models/extractor_ner/train.spacy"

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = "models/extractor_ner/val.spacy"
"""
    (output / "config.cfg").write_text(cfg)

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main(train_jsonl: str, val_jsonl: str, out_dir: str):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to .spacy
    blank = spacy.blank("en")
    for lbl in ALL_LABELS:
        blank.add_pipe("ner", last=True) if "ner" not in blank.pipe_names else None
        blank.get_pipe("ner").add_label(lbl)

    print("ℹ Converting JSONL → DocBin …")
    jsonl_to_docbin(train_jsonl, blank).to_disk(out_path / "train.spacy")
    jsonl_to_docbin(val_jsonl,   blank).to_disk(out_path / "val.spacy")

    # Step 2: Write config
    print("ℹ Writing config …")
    make_config(out_path)

    # Step 3: Train
    print("ℹ Training spaCy NER …")
    cmd = [
        "python","-m","spacy","train",
        str(out_path / "config.cfg"),
        "--output", str(out_path),
        "--paths.train", str(out_path / "train.spacy"),
        "--paths.dev",   str(out_path / "val.spacy"),
        "--gpu-id","-1"          # change to 0 for GPU
    ]
    subprocess.run(cmd, check=True)
    print("✅ Training complete. Model saved to", out_path)


# CLI -----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/synth/train.jsonl")
    ap.add_argument("--val",   default="data/synth/val.jsonl")
    ap.add_argument("--out",   default="models/extractor_ner")
    args = ap.parse_args()
    main(args.train, args.val, args.out)
