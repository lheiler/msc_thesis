# --------------------------------------------------
#  Latent feature extraction methods
# --------------------------------------------------
from latent_extraction.cortico_thalamic import fit_ctm_from_raw
from torch.utils.data import DataLoader
from latent_extraction.c22 import extract_c22, extract_c22_psd
from latent_extraction.ctm_nn.nn_ctm_parameters import ParameterRegressor

from latent_extraction.ctm_nn.nn_ctm_parameters import infer_latent_parameters

import mne
import numpy as np
import os
import json
import torch
# For CwA-T we will lazily import to avoid unnecessary dependency cost when
# other methods are used.  Import inside functions below.

def clean_x(x):
    if 'A1' in x.ch_names:
        x.drop_channels(['A1'])
    if 'A2' in x.ch_names:
        x.drop_channels(['A2'])
    # specifically pick the 19 EEG channels by name
    eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                    'Cz', 'Pz', 'Fz']    
    x.pick_channels(eeg_channels)
    return x

def extract_latent_features(data: DataLoader, batch_size, method, save_path=""):
    """
    Extract latent features from the EEG data and optionally save them.
    """
    latent_features = []
    model = None

    # ✅ Truncate file at the start (overwrite if it exists)
    if save_path:
        with open(save_path, "w") as f:
            pass  # Clear file if exists

    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )

    # ------------------------------------------------------------------
    # Pre-load heavy models that will be reused across the dataset
    # ------------------------------------------------------------------
    if method in {"cwat", "CwA-T", "cwa_t"}:
        from latent_extraction.cwat_autoencoder import get_cwat_model, extract_cwat

        _ = get_cwat_model(device=device)  # warm-up so subsequent calls are cheap

    else:
        extract_cwat = None  # type: ignore

    # --------------------------------------------------------------
    # Pre-load **neural CTM encoder** once if requested
    # --------------------------------------------------------------
    if method in {"ctm_nn"}:
        # Load weights (path can be overridden via env var)
        model_path = "/homes/lrh24/thesis/testing/amore/models/regressor_1e-4-8.pt"
        model = ParameterRegressor()
        model.load_state_dict(torch.load(model_path)["model_state"])
        model.to(device)
    else:
        model = None

    for x, g, a, ab in data:
        x = clean_x(x)
        if method == "ctm":
            latent_feature = fit_ctm_from_raw(x, as_vector=True)
        elif method == "ctm_nn":
            # Neural CTM encoder → vector (19×8 -> flattened)
            try:
                latent_feature = infer_latent_parameters(model, x, device=device)
            except Exception as e:
                print(f"⚠️  Failed to extract CTM-NN features: {e}")
                latent_feature = None
        elif method == "c22":
            latent_feature = extract_c22(x)
        elif method == "c22_psd":
            latent_feature = extract_c22_psd(x)
        elif method in {"cwat", "CwA-T", "cwa_t"}:
            # extract_cwat was imported above only if needed
            latent_feature = extract_cwat(x, device=device)[0]  # type: ignore
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if latent_feature is None:
            continue

        # Ensure output is a float32 tensor for downstream .detach() usage
        latent_feature = torch.as_tensor(latent_feature, dtype=torch.float32)

        # Serialize safely
        record = (
            latent_feature.tolist() if hasattr(latent_feature, 'tolist') else latent_feature,
            int(g.item()) if hasattr(g, 'item') else int(g),
            int(a.item()) if hasattr(a, 'item') else int(a),
            int(ab.item()) if hasattr(ab, 'item') else int(ab)
        )

        # Append to the file
        if save_path:
            with open(save_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        latent_features.append((latent_feature, g, a, ab))

    return DataLoader(latent_features, batch_size=batch_size, shuffle=False)
