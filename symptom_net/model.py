# symptom_net/model.py

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class SymptomNet(nn.Module):
    def __init__(self, leaf_cnt, meta_dim, enc_name="distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(enc_name)
        self.enc = AutoModel.from_pretrained(enc_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.enc.config.hidden_size + meta_dim, leaf_cnt)
        self.regressor = nn.Linear(self.enc.config.hidden_size + meta_dim, 1)

    def forward(self, texts, meta):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if next(self.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            meta = meta.cuda()
        out = self.enc(**inputs).last_hidden_state[:, 0]  # CLS token
        combined = torch.cat([out, meta], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined), self.regressor(combined).squeeze(-1)
