from Latent_Extraction.cortico_thalamic import fit_ctm_from_raw 
from torch.utils.data import DataLoader
from Latent_Extraction.c22 import extract_c22
from Latent_Extraction.c22 import extract_c22_psd
from Latent_Extraction.AE.convAE import Conv1DAutoencoder
from Latent_Extraction.AE.extract_z import extract_z
import os
import json
import torch



def extract_latent_features(data: DataLoader, batch_size, method, save_path=""):
    """
    Extract latent features from the EEG data and optionally save them.
    """
    latent_features = []
    model = None

    # ✅ Truncate file at the start (overwrite if it exists)
    if save_path:
        with open(save_path, "w") as f:
            pass  # This will clear the file

    if method == "AE":
        model = Conv1DAutoencoder(n_channels=19, fixed_len=len(data[0][0]), latent_dim=16)
        model.load_state_dict(torch.load("/rds/general/user/lrh24/home/thesis/code/Latent_Extraction/AE/conv1d_autoencoder.pth"))

    for x, g, a, ab in data:
        if method == "ctm":
            latent_feature = fit_ctm_from_raw(x)
        elif method == "c22":
            latent_feature = extract_c22(x)
        elif method == "c22_psd":
            latent_feature = extract_c22_psd(x)
        elif method == "AE":
            latent_feature = extract_z(model, x)
        else:
            raise ValueError(f"Unknown method: {method}")

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
