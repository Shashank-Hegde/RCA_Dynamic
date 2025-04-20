import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class SymptomNet(nn.Module):
    def __init__(self, num_labels: int, meta_dim: int):
        super().__init__()
        self.enc = AutoModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.linear = nn.Sequential(
            nn.Linear(meta_dim + self.enc.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_labels)
        self.regressor = nn.Linear(256, 1)

    def forward(self, texts, meta_vec):
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                return_tensors="pt").to(meta_vec.device)
        enc_out = self.enc(**tokens).last_hidden_state[:, 0]
        z = torch.cat([enc_out, meta_vec], dim=-1)
        h = self.linear(z)
        return self.classifier(h), torch.sigmoid(self.regressor(h))
