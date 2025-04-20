# symptom_net/utils.py

import torch
from typing import Any
from symptom_net.constants import CANON_KEYS

def dict_to_vec(d: dict[str, Any]) -> torch.Tensor:
    """
    Convert a dictionary of extracted features to a fixed-length vector.
    Only keys in CANON_KEYS are considered, in order.
    """
    vec = []
    for k in CANON_KEYS:
        val = d.get(k)
        if isinstance(val, bool):
            vec.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            vec.append(float(val))
        elif isinstance(val, str):
            vec.append(1.0 if val.strip() else 0.0)  # presence indicator
        elif isinstance(val, dict):
            vec.append(float(len(val)))  # number of entries in submap
        else:
            vec.append(0.0)
    return torch.tensor(vec, dtype=torch.float)
