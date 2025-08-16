# --------------------------------------------------
#  Latent feature extraction methods
# --------------------------------------------------
from latent_extraction.cortico_thalamic import fit_ctm_average_from_raw
from latent_extraction.cortico_thalamic import fit_ctm_per_channel_from_raw
from latent_extraction.eeg2rep.extract_eeg2rep import get_eeg2rep_model as _get_eeg2rep_model  # type: ignore
from latent_extraction.eeg2rep.extract_eeg2rep import extract_eeg2rep as _extract_eeg2rep  # type: ignore
from torch.utils.data import DataLoader
from latent_extraction.psd_ae.psd_ae import get_psd_ae_model
from utils.util import (
    clean_raw_eeg,
    select_device,
    ensure_float32_tensor,
    make_latent_record,
    truncate_file,
    append_jsonl,
    method_input_format,
    INPUT_FORMAT_RAW_19,
    INPUT_FORMAT_PSD_AVG,
    INPUT_FORMAT_PSD_PER_CHANNEL,
    compute_avg_psd_from_raw,
    compute_psd_from_raw,
)
from latent_extraction.c22 import extract_c22, extract_c22_from_psd
from latent_extraction.ctm_nn.nn_ctm_parameters import ParameterRegressor
from pathlib import Path
import importlib.util

from latent_extraction.ctm_nn.nn_ctm_parameters import infer_latent_parameters
from latent_extraction.hopf import fit_hopf_from_raw

import mne
import numpy as np
import os
import json
import torch
#
import os



def extract_latent_features(data: DataLoader, batch_size, method, save_path=""):
    """
    Extract latent features from the EEG data and optionally save them.
    """
    latent_features = []
    model = None

    # ✅ Truncate file at the start (overwrite if it exists)
    if save_path:
        truncate_file(save_path)

    device = select_device()


    # --------------------------------------------------------------
    # Pre-load models once if requested
    # --------------------------------------------------------------
    if method in {"ctm_nn"}:
        # Load weights (path can be overridden via env var)
        model_path = "/homes/lrh24/thesis/testing/amore/models/regressor_1e-4-8.pt"
        model = ParameterRegressor()
        model.load_state_dict(torch.load(model_path)["model_state"])
        model.to(device)
    elif method in {"eegnet"}:
        # Hyphenated directory requires path-based import
        this_dir = Path(__file__).resolve().parent
        infer_path = this_dir / "EEGNet-AE" / "infer.py"
        spec = importlib.util.spec_from_file_location("eegnet_ae_infer", str(infer_path))
        assert spec and spec.loader, f"Could not load spec for {infer_path}"
        eegnet_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eegnet_mod)
        # Load model once; keep extractor for later calls
        model = eegnet_mod.get_eegnet_ae_model(device=device)
        extract_eegnet_ae = eegnet_mod.extract_eegnet_ae  # type: ignore
    elif method in {"eeg2rep"}:
        model = _get_eeg2rep_model(device=device, checkpoint_path=os.environ.get("EEG2REP_CKPT"))
        extract_eeg2rep_vec = _extract_eeg2rep  # type: ignore
    elif method in {"pca_psd"}:
        # Lazy-load PCA runtime and prepare PSD parameters
        from latent_extraction.pca.pca import FrozenPCATorch  # type: ignore
        pca_model_path = "/homes/lrh24/thesis/code/latent_extraction/pca/models/pca_avg_psd_k8.npz"
        model = FrozenPCATorch(pca_model_path, device=device)
    elif method in {"psd_ae", "psd_ae_avg", "psd_ae_channel", "psd_ae_ch"}:
        model = get_psd_ae_model(device=device)
    else:
        model = None

    if model is None:
        raise ValueError(f"Unknown method: {method}")
    
    for x, g, a, ab in data:
        x = clean_raw_eeg(x)
        
        if method == "ctm_avg":
            latent_feature = fit_ctm_average_from_raw(x)
        elif method == "ctm_per_channel":
            latent_feature = fit_ctm_per_channel_from_raw(x)
        elif method == "ctm_nn":
            latent_feature = infer_latent_parameters(model, x, device=device)
        elif method == "c22":
            latent_feature = extract_c22(x)
        elif method == "c22_psd":
            latent_feature = extract_c22_from_psd(x)
        elif method in {"eegnet"}:
            latent_feature = extract_eegnet_ae(x, device=device, model=model)  # type: ignore
        elif method in {"eeg2rep"}:
            latent_feature = extract_eeg2rep_vec(x, device=device, model=model, segment_len_sec=10, target_sfreq=128.0)  
        elif method in {"pca"}:
            psd = compute_psd_from_raw(x, calculate_average=True)
            latent_feature = model.transform_vec(psd)  
        elif method in {"hopf"}:
            latent_feature = fit_hopf_from_raw(x)
        elif method in {"psd_ae", "psd_ae_avg"}:
            latent_feature = _extract_psd_ae_avg(x, device=device, model=model)
        elif method in {"psd_ae_channel", "psd_ae_ch"}:
            latent_feature = _extract_psd_ae_channel(x, device=device, model=model)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if latent_feature is None:
            raise ValueError(f"No latent feature extracted for {method}")

        # Ensure output is a float32 tensor for downstream .detach() usage
        latent_feature = ensure_float32_tensor(latent_feature)

        # Serialize safely
        record = make_latent_record(latent_feature, g, a, ab)

        # Append to the file
        if save_path:
            append_jsonl(save_path, record)

        latent_features.append((latent_feature, g, a, ab))

    return DataLoader(latent_features, batch_size=batch_size, shuffle=False)
