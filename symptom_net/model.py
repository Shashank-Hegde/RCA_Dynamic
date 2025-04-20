# symptom_net/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class SymptomNet(nn.Module):
    def __init__(self, num_labels: int, meta_dim: int, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.enc.config.hidden_size + meta_dim
        self.classifier = nn.Linear(self.hidden_dim, num_labels)
        self.regressor = nn.Linear(self.hidden_dim, 1)

    def forward(self, texts, meta_vec):
        inputs = self.enc.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.enc(**inputs).last_hidden_state[:, 0]  # [CLS] token
        combined = torch.cat([outputs, meta_vec], dim=1)
        logits = self.classifier(combined)
        risk = self.regressor(combined)
        return logits, risk
