# symptom_net/model.py

import torch
import torch.nn as nn

class SymptomNet(nn.Module):
    def __init__(self, output_dim: int, meta_dim: int = 100):
        super().__init__()
        self.text_encoder = nn.EmbeddingBag(1000, 128, sparse=True)  # Placeholder
        self.linear = nn.Sequential(
            nn.Linear(meta_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256, output_dim)
        self.regressor = nn.Linear(256, 1)

    def forward(self, texts, meta):
        x = self.linear(meta)
        logits = self.classifier(x)
        risk = self.regressor(x)
        return logits, risk
