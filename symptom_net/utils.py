import numpy as np, torch
from symptom_net.constants import CANON_KEYS

_CAT = ["sex","region","ethnicity"]
_FLOAT = ["age", "max_temp"]

_CAT2IDX = {val:i for i,val in enumerate(sorted(set(_CAT)))}

def dict_to_vec(d):
    vec = []
    for k in _FLOAT:
        vec.append(float(d.get(k,0))/100)
    for k in _CAT:
        one = np.zeros(len(_CAT2IDX));
        v = d.get(k)
        if v in _CAT2IDX: one[_CAT2IDX[v]] = 1
        vec.extend(one)
    return torch.tensor(vec, dtype=torch.float32)
