# scripts/demo_predict_paths.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from symptom_net.extractor import extract
from symptom_net.utils import dict_to_vec
from symptom_net.model import SymptomNet
from symptom_net.constants import CANON_KEYS
import torch, yaml

ONTO = yaml.safe_load(open("ontology/v1.yaml"))
ID2NODE = {n["id"]: n for n in ONTO}
LEAVES = [n for n in ONTO if not any(c["parent_id"] == n["id"] for c in ONTO)]
leaf2idx = {l["id"]: i for i, l in enumerate(LEAVES)}
idx2leaf = {i: l["id"] for i, l in enumerate(LEAVES)}

extracted = extract(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--text' else input("Enter text: "), {})
meta_vec = dict_to_vec(extracted).unsqueeze(0)
model = SymptomNet(len(LEAVES), meta_dim=meta_vec.shape[1])
model.load_state_dict(torch.load("models/symptom_net.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    logits, risk = model([""], meta_vec)
    probs = torch.sigmoid(logits[0]).tolist()
    pred_idx = int(torch.tensor(probs).argmax())
    pred_leaf = idx2leaf[pred_idx]

    def iter_path(leaf_id):
        path = []
        while leaf_id:
            path.append(leaf_id)
            leaf_id = ID2NODE[leaf_id].get("parent_id")
        return list(reversed(path))

    print("\n===============================")
    print(f"Extracted: {extracted}\n")
    print(f"Predicted Leaf: {pred_leaf}")
    print(f"Symptom Path : {' → '.join(iter_path(pred_leaf))}")
    print(f"Risk Score    : {risk.item():.3f}\n")

    print("Top 5 leaf predictions:")
    topk = torch.topk(torch.tensor(probs), 5)
    for rank, (score, idx) in enumerate(zip(topk.values, topk.indices)):
        print(f"  #{rank+1}: {idx2leaf[idx.item()]}  ({score.item():.3f})")
