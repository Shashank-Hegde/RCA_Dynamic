import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 🔥 this is key

import json, yaml, torch, argparse, pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch import nn

from symptom_net.model import SymptomNet
from symptom_net.utils import dict_to_vec
from symptom_net.constants import CANON_KEYS

class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, leaf2idx):
        self.rows = [json.loads(l) for l in open(jsonl_path)]
        self.l2i = leaf2idx

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        y = torch.zeros(len(self.l2i))
        y[self.l2i[r["label_leaf_id"]]] = 1
        return {
            "text": r["text"],
            "meta": dict_to_vec(r["extracted"]),
            "y": y,
            "risk": torch.tensor(r.get("risk", 0.0), dtype=torch.float32)
        }

def collate(batch):
    return {
        "text": [b["text"] for b in batch],
        "meta": torch.stack([b["meta"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "risk": torch.stack([b["risk"] for b in batch])
    }

class Lit(pl.LightningModule):
    def __init__(self, enc, leaf_cnt, meta_dim, lr):
        super().__init__()
        self.net = SymptomNet(leaf_cnt, meta_dim, enc)
        self.lr = lr

    def forward(self, batch):
        return self.net(batch["text"], batch["meta"])

    def step(self, batch):
        logits, risk = self(batch)
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["y"])
        loss2 = torch.nn.functional.mse_loss(risk, batch["risk"])
        return loss1 + 0.2 * loss2

    def training_step(self, batch, _):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        self.log("val_loss", self.step(batch))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/synth/train.jsonl")
    p.add_argument("--val", default="data/synth/val.jsonl")
    p.add_argument("--onto", default="ontology/v1.yaml")
    p.add_argument("--enc", default="distilbert-base-uncased")
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    onto = yaml.safe_load(open(args.onto))
    leaves = [n for n in onto if not any(c["parent_id"] == n["id"] for c in onto)]
    l2i = {l["id"]: i for i, l in enumerate(leaves)}
    meta_dim = dict_to_vec({}).shape[0]

    tr_ds = JsonDataset(args.train, l2i)
    va_ds = JsonDataset(args.val, l2i)
    dm = lambda ds: DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=collate)

    lit = Lit(args.enc, len(leaves), meta_dim, args.lr)
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed", max_epochs=args.epochs)
    trainer.fit(lit, dm(tr_ds), dm(va_ds))

    Path("models").mkdir(exist_ok=True)
    #torch.save(lit.net.state_dict(), "models/symptom_net.pt")
    torch.save(lit.net, "models/symptom_net.pt")  # Save the entire model
    print("✅ models/symptom_net.pt saved")

if __name__ == "__main__":
    main()
