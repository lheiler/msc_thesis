from latent_extraction.cortico_thalamic import fit_ctm_from_raw
from torch.utils.data import DataLoader
from latent_extraction.c22 import extract_c22, extract_c22_psd
import os
import json
import torch
# For CwA-T we will lazily import to avoid unnecessary dependency cost when
# other methods are used.  Import inside functions below.


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
    
    for x, g, a, ab in data:
        if method == "ctm":
            latent_feature = fit_ctm_from_raw(x, as_vector=True)
        elif method == "c22":
            latent_feature = extract_c22(x)
        elif method == "c22_psd":
            latent_feature = extract_c22_psd(x)
        elif method in {"cwat", "CwA-T", "cwa_t"}:
            # extract_cwat was imported above only if needed
            latent_feature = extract_cwat(x, device=device)  # type: ignore
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
