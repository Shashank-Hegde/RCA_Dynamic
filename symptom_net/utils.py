# symptom_net/utils.py
import torch
import numpy as np
from symptom_net.constants import CANON_KEYS

def dict_to_vec(d: dict) -> torch.Tensor:
    """
    Convert extracted dict {CANON_KEY: value} → fixed-length vector.
    Handles binary, categorical, and mapped keys.
    Missing keys are zero-filled.
    """

    def encode_value(key, val):
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        elif isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, str):
            return float(len(val)) / 100.0  # Normalize string length
        elif isinstance(val, dict):
            # For maps like symptom_duration_map: sum normalized durations
            return float(len(val)) / 10.0
        elif val is None:
            return 0.0
        return 0.0

    vec = []
    for k in CANON_KEYS:
        v = d.get(k)
        vec.append(encode_value(k, v))

    return torch.tensor(vec, dtype=torch.float)
