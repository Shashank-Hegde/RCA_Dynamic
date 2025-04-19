import yaml, torch
from dataclasses import dataclass, field
from symptom_net.extractor import extract
from symptom_net.utils import dict_to_vec, iter_path
from symptom_net.model import SymptomNet

ONTO = yaml.safe_load(open("ontology/v1.yaml"))
ID2N = {n["id"]: n for n in ONTO}
LEAVES = [n for n in ONTO if not any(c["parent_id"]==n["id"] for c in ONTO)]
leaf2idx = {l["id"]: i for i, l in enumerate(LEAVES)}
idx2leaf = {i: l["id"] for i, l in enumerate(LEAVES)}

meta_dim = dict_to_vec({}).shape[0]
net = SymptomNet(len(LEAVES), meta_dim)
net.load_state_dict(torch.load("models/symptom_net.pt", map_location="cpu"))
net.eval()

ADDITIONAL = [
    ("May I know your age?", "age"),
    ("Do you have any chronic illnesses?", "chronic_illnesses"),
    ("Which city or locality do you live in?", "location")
]

@dataclass
class Session:
    extracted: dict = field(default_factory=dict)
    asked: set = field(default_factory=set)
    path: list = field(default_factory=list)

def classify(text, meta):
    with torch.no_grad():
        logits, risk = net([text], meta)
    return torch.sigmoid(logits)[0], risk.item()

def next_q(sess):
    for nid in sess.path:
        for f in ID2N[nid]["followups"]:
            if f["q"] in sess.asked: continue
            if not set(f["asks_for"]).issubset(sess.extracted):
                sess.asked.add(f["q"]); return f["q"]
    for q, var in ADDITIONAL:
        if var not in sess.extracted and q not in sess.asked:
            sess.asked.add(q); return q
    return "Thank you, I have all the information I need."

def chat():
    s = Session()
    while True:
        user = input("Patient: ")
        s.extracted = extract(user, s.extracted)
        meta = dict_to_vec(s.extracted).unsqueeze(0)
        probs, risk = classify(user, meta)
        leaf = idx2leaf[int(probs.argmax())]
        s.path = list(iter_path(leaf, ID2N))
        print("Assistant:", next_q(s), f"[risk={risk:.2f}]")

if __name__ == "__main__":
    chat()