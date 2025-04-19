#!/usr/bin/env python
"""Auto‑label raw narratives → enriched JSONL"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, yaml, torch, argparse
from pathlib import Path
from tqdm import tqdm
from symptom_net.extractor import extract
from symptom_net.utils import dict_to_vec
from symptom_net.model import SymptomNet

p = argparse.ArgumentParser()
p.add_argument("--infile", default="data/new_batch/unlabeled.jsonl")
p.add_argument("--outfile", default="data/new_batch/with_predictions.jsonl")
p.add_argument("--onto", default="ontology/v1.yaml")
args = p.parse_args()

# ontology
ONTO = yaml.safe_load(open(args.onto))
LEAVES = [n for n in ONTO if not any(c["parent_id"]==n["id"] for c in ONTO)]
leaf2idx = {l["id"]: i for i, l in enumerate(LEAVES)}
idx2leaf = {i: l["id"] for i, l in enumerate(LEAVES)}

# model
meta_dim = dict_to_vec({}).shape[0]
model = SymptomNet(len(LEAVES), meta_dim)
model.load_state_dict(torch.load("models/symptom_net.pt", map_location="cpu"))
model.eval()

Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
with open(args.infile) as fin, open(args.outfile, "w") as fout:
    for line in tqdm(fin, desc="Auto‑labelling"):
        j = json.loads(line)
        ext = extract(j["text"], {})
        meta = dict_to_vec(ext).unsqueeze(0)
        with torch.no_grad():
            logits, r = model([j["text"]], meta)
            leaf = idx2leaf[int(torch.sigmoid(logits[0]).argmax())]
            risk = round(r.item(), 3)
        j.update({"extracted": ext, "label_leaf_id": leaf, "risk": risk})
        fout.write(json.dumps(j)+"\n")
print("✅ written", args.outfile)
