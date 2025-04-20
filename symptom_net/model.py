# symptom_net/model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SymptomNet(nn.Module):
    def __init__(self, num_labels, meta_dim):
        super().__init__()
        self.enc = AutoModel.from_pretrained("distilbert-base-uncased")
        self.linear = nn.Sequential(
            nn.Linear(self.enc.config.hidden_size + meta_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )
        self.risk = nn.Sequential(
            nn.Linear(self.enc.config.hidden_size + meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, texts, metas):
        inputs = self.enc.batch_encode_plus(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        h = self.enc(**inputs).last_hidden_state[:, 0]  # CLS token
        x = torch.cat([h, metas], dim=1)
        return self.linear(x), self.risk(x)
