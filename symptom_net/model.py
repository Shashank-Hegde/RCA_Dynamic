# symptom_net/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class SymptomNet(nn.Module):
    def __init__(self, num_labels: int, meta_dim: int = 64):  # or actual meta_dim
        super().__init__()
        input_dim = 768 + meta_dim
        self.classifier = nn.Linear(input_dim, num_labels)
        self.regressor = nn.Linear(input_dim, 1)

    def forward(self, text_embeddings, meta_vec):
        if isinstance(text_embeddings, list):
            # Replace with your CLS token from model if available
            text_vec = torch.randn(1, 768)
        else:
            text_vec = text_embeddings
        x = torch.cat([text_vec, meta_vec], dim=1)
        return self.classifier(x), self.regressor(x)
