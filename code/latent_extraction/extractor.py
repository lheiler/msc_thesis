from latent_extraction.cortico_thalamic import fit_ctm_from_raw
from torch.utils.data import DataLoader
from latent_extraction.c22 import extract_c22, extract_c22_psd
from latent_extraction.AE.convAE import Conv1DAutoencoder, EEGAutoEncoder
from latent_extraction.AE.extract_z import extract_z
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if method == "AE":
        model = EEGAutoEncoder(chans=19, fixed_len=7680, latent_dim=64)
        # Resolve checkpoint path relative to this file to remain robust to folder renaming
        ckpt_path = os.path.join(os.path.dirname(__file__), "AE", "conv1d_autoencoder.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    for x, g, a, ab in data:
        if method == "ctm":
            latent_feature = fit_ctm_from_raw(x)
        elif method == "c22":
            latent_feature = extract_c22(x)
        elif method == "c22_psd":
            latent_feature = extract_c22_psd(x)
        elif method == "AE":
            model_device = next(model.parameters()).device
            latent_feature = extract_z(model, x, device=device)
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
