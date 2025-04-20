# symptom_net/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class SymptomNet(nn.Module):
    def __init__(self, num_labels, meta_dim):
        super().__init__()
        self.enc = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_dim = self.enc.config.hidden_size + meta_dim
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, texts, metas):
        inputs = self.enc.batch_encode_plus(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        h = self.enc(**inputs).last_hidden_state[:, 0]  # CLS token
        x = torch.cat([h, metas], dim=1)
        return self.classifier(x), torch.sigmoid(self.regressor(x))
