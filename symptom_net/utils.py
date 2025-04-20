# symptom_net/utils.py

import torch
from typing import Any
from symptom_net.constants import CANON_KEYS

def dict_to_vec(d: dict[str, Any]) -> torch.Tensor:
    vec = []
    for key in CANON_KEYS:
        val = d.get(key)
        if isinstance(val, bool):
            vec.append(float(val))
        elif isinstance(val, (int, float)):
            vec.append(float(val))
        elif isinstance(val, str):
            vec.append(float(len(val)))
        elif isinstance(val, dict):
            vec.append(float(len(val)))
        else:
            vec.append(0.0)
    return torch.tensor(vec, dtype=torch.float)
