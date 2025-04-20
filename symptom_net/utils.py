import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 🔥 this is key

import torch
from typing import Any
from symptom_net.constants import CANON_KEYS

def dict_to_vec(d: dict[str, Any]) -> torch.Tensor:
    vec = []
    for k in CANON_KEYS:
        val = d.get(k)
        if isinstance(val, bool):
            vec.append(float(val))
        elif isinstance(val, (int, float)):
            vec.append(float(val))
        elif isinstance(val, str):
            vec.append(1.0)
        elif isinstance(val, dict):
            vec.append(float(len(val)))
        else:
            vec.append(0.0)
    return torch.tensor(vec, dtype=torch.float)
