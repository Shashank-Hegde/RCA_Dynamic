#!/usr/bin/env python

import spacy
import json
import pathlib
import argparse
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch

ALL_LABELS = [
    "AGE", "SEX", "ETHNICITY", "LOCATION", "REGION", "SOCIOECONOMIC_STATUS",
    "PAST_CONDITIONS", "SURGERIES", "HOSPITALISATIONS", "CHRONIC_ILLNESSES",
    "MEDICATION_HISTORY", "IMMUNISATION_STATUS", "ALLERGIES",
    "FAMILY_HISTORY", "DIET", "PHYSICAL_ACTIVITY", "SLEEP_PATTERN",
    "ALCOHOL", "TOBACCO", "MENTAL_HEALTH", "WORK_STRESS",
    "ENVIRONMENTAL_EXPOSURE", "HOUSING", "CLEAN_WATER", "OCCUPATION"
]

def jsonl_to_examples(path, nlp):
    examples = []
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
        examples.append(Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}))
    return examples

def train_ner_model(train_path, val_path, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for lbl in ALL_LABELS:
        ner.add_label(lbl)

    train_data = jsonl_to_examples(train_path, nlp)
    val_data = jsonl_to_examples(val_path, nlp)

    optimizer = nlp.begin_training()
    for epoch in range(10):
        losses = {}
        batches = minibatch(train_data, size=8)
        for batch in batches:
            nlp.update(batch, drop=0.3, losses=losses)
        print(f"Epoch {epoch+1} Losses: {losses}")

    nlp.to_disk(out_dir / "model")
    print(f"✔ Model saved to {out_dir / 'model'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/synth/train.jsonl")
    p.add_argument("--val", default="data/synth/val.jsonl")
    p.add_argument("--out", default="models/extractor_ner")
    args = p.parse_args()
    train_ner_model(args.train, args.val, args.out)
