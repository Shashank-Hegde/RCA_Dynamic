# ✅ scripts/demo_predict_paths.py
"""
Simple demo to predict full paths for multiple patient inputs
and show model logits (trace mode) for every symptom leaf.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from symptom_net.extractor import extract
from symptom_net.utils import dict_to_vec
from symptom_net.model import SymptomNet
from symptom_net.constants import CANON_KEYS
import torch, yaml

# Load ontology + leaf info
ONTO = yaml.safe_load(open("ontology/v1.yaml"))
ID2NODE = {n["id"]: n for n in ONTO}
LEAVES = [n for n in ONTO if not any(c["parent_id"] == n["id"] for c in ONTO)]
leaf2idx = {l["id"]: i for i, l in enumerate(LEAVES)}
idx2leaf = {i: l["id"] for i, l in enumerate(LEAVES)}

# Sample inputs
inputs = [
    "Hi Doctor, I’ve had a persistent fever and wet cough for 3 weeks. Recently, I noticed some blood in the sputum.",
    "Good evening, I have sharp stomach pain in the upper right side after fatty meals, and feel nauseous.",
    "Hello, I've had a mild headache for 2 days, mostly around my temples, but no nausea or vision issues."
]

def iter_path(leaf_id):
    """Return full symptom path from leaf to root"""
    path = []
    while leaf_id:
        path.append(leaf_id)
        leaf_id = ID2NODE[leaf_id]["parent_id"]
    return list(reversed(path))

for i, text in enumerate(inputs):
    print(f"\n===============================\nINPUT #{i+1}: {text}\n")
    extracted = extract(text, {})
    meta_vec = dict_to_vec(extracted).unsqueeze(0)
    model = SymptomNet(len(LEAVES), meta_dim=meta_vec.shape[1])
    model.load_state_dict(torch.load("models/symptom_net.pt", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits, risk = model([text], meta_vec)
        probs = torch.sigmoid(logits[0]).tolist()
        pred_idx = int(torch.tensor(probs).argmax())
        pred_leaf = idx2leaf[pred_idx]
        path = iter_path(pred_leaf)

    print(f"Predicted Leaf: {pred_leaf}")
    print(f"Symptom Path : {' → '.join(path)}")
    print(f"Risk Score    : {risk.item():.3f}\n")

    print("Top 5 leaf predictions:")
    topk = torch.topk(torch.tensor(probs), 5)
    for rank, (score, idx) in enumerate(zip(topk.values, topk.indices)):
        print(f"  #{rank+1}: {idx2leaf[idx.item()]}  ({score.item():.3f})")
