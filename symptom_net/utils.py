import torch

def dict_to_vec(d):
    # Same 64 keys/features used during training
    KEYS = [f"f{i}" for i in range(64)]  # placeholder names

    vec = []
    for k in KEYS:
        v = d.get(k, 0.0)
        vec.append(float(v))
    return torch.tensor(vec).float()
