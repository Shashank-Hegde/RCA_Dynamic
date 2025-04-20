# symptom_net/utils.py
import torch
import numpy as np
from symptom_net.constants import CANON_KEYS

def dict_to_vec(d):
    """Convert canonical symptom dict to binary+meta vector."""
    vec = []
    for key in CANON_KEYS:
        v = d.get(key)
        if isinstance(v, bool):
            vec.append(int(v))
        elif isinstance(v, (int, float)):
            vec.append(float(v))
        elif v is None:
            vec.append(0.0)
        else:
            vec.append(1.0)
    return torch.tensor(vec, dtype=torch.float32)
