import re
import json
import torch
from torch.utils.data import DataLoader


def load_latent_parameters_array(file_path: str, batch_size: int = 32):
    """Read a .json lines file where each line is a JSON list:
        [[float, ...], gender, age, abnormal]
    and return a DataLoader with tensor fields.
    """
    latent_params = []
    file_path = file_path + ".json" if not file_path.endswith(".json") else file_path

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                vec_or_dict = entry[0]
                # Handle both list-of-floats and dict-of-name->value formats
                if isinstance(vec_or_dict, dict):
                    # Sort keys to ensure consistent ordering across rows
                    ordered_vals = [float(vec_or_dict[k]) for k in sorted(vec_or_dict.keys())]
                    latent_vec = torch.tensor(ordered_vals, dtype=torch.float32)
                else:
                    latent_vec = torch.tensor(vec_or_dict, dtype=torch.float32)
                g = torch.tensor(entry[1], dtype=torch.float32)
                a = torch.tensor(entry[2], dtype=torch.float32)
                ab = torch.tensor(entry[3], dtype=torch.float32)
                latent_params.append((latent_vec, g, a, ab))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    print(f"Loaded {len(latent_params)} latent parameters from {file_path}")
    return DataLoader(latent_params, batch_size=batch_size, shuffle=False) 