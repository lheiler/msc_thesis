# --------------------------------------------------
#  Latent feature extraction methods
# --------------------------------------------------
from latent_extraction.cortico_thalamic import fit_ctm_average_from_raw
from latent_extraction.cortico_thalamic import fit_ctm_per_channel_from_raw
from torch.utils.data import DataLoader
from latent_extraction.psd_ae.psd_ae import get_psd_ae_model, extract_psd_ae_avg, extract_psd_ae_channel
from latent_extraction.wong_wang import fit_wong_wang_average_from_raw, fit_wong_wang_per_channel_from_raw
from latent_extraction.EEGNet_AE.infer import get_eegnet_ae_model, extract_eegnet_ae
import latent_extraction.hopf as hopf
from latent_extraction.jansen_rit import fit_jr_average_from_raw, fit_jr_per_channel_from_raw
from utils.util import (
    select_device,
    ensure_float32_tensor,
    make_latent_record,
    truncate_file,
    append_jsonl,
)
from latent_extraction.c22 import extract_c22
from latent_extraction.ctm_nn.nn_ctm_parameters import ParameterRegressor
from latent_extraction.ctm_nn.nn_ctm_parameters import infer_latent_parameters
from latent_extraction.pca.pca import extract_pca_from_raw, FrozenPCATorch
import numpy as np
import os
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

PARALLELIZABLE_METHODS = {
    "ctm_cma_pc", "ctm_cma_avg",
    "jr_pc", "jr_avg",
    "wong_wang_pc", "wong_wang_avg",
    "hopf_pc", "hopf_avg", "c22"
}

def _run_single_parallel(method: str, x):
    if method == "ctm_cma_pc":
        return fit_ctm_per_channel_from_raw(x)
    if method == "ctm_cma_avg":
        return fit_ctm_average_from_raw(x)
    if method == "jr_pc":
        return fit_jr_per_channel_from_raw(x)
    if method == "jr_avg":
        return fit_jr_average_from_raw(x)
    if method == "wong_wang_pc":
        return fit_wong_wang_per_channel_from_raw(x)
    if method == "wong_wang_avg":
        return fit_wong_wang_average_from_raw(x)
    if method == "hopf_pc":
        return hopf.fit_hopf_from_raw(x, per_channel=True)
    if method == "hopf_avg":
        return hopf.fit_hopf_from_raw(x, per_channel=False)
    if method == "c22":
        return extract_c22(x)
    raise ValueError(f"Method not supported for parallel execution: {method}")

def extract_latent_features(data: DataLoader, batch_size, method, save_path="", n_workers: int = 64):
    """
    Extract latent features from the EEG data and optionally save them.
    """
    latent_features = []
    model = None
    if save_path:
        truncate_file(save_path)
    device = select_device()

    # If requested, run item-level parallel extraction for supported methods
    if n_workers and n_workers > 1 and method in PARALLELIZABLE_METHODS:
        # Materialize the data loader so we can dispatch jobs and keep metadata
        items = list(data)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_run_single_parallel, method, item[0]) for item in items]
            # Gather results in submission order to preserve dataset order
            for (x, g, a, ab, sample_id), fut in zip(items, futures):
                latent_feature = fut.result()
                if latent_feature is None:
                    raise ValueError(f"No latent feature extracted for {method}")
                if np.ndim(latent_feature) != 1:
                    raise ValueError(f"Latent feature must be 1D, got {np.ndim(latent_feature)}D")
                latent_feature = ensure_float32_tensor(latent_feature)
                record = make_latent_record(latent_feature, g, a, ab, sample_id)
                if save_path:
                    append_jsonl(save_path, record)
                latent_features.append((latent_feature, g, a, ab))
        return DataLoader(latent_features, batch_size=batch_size, shuffle=False)

    if method == "ctm_nn_pc" or method == "ctm_nn_avg":
        model = ParameterRegressor().to(device)
        state = torch.load("/rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore/models/regressor.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"]) 
    elif method == "eegnet":
        model = get_eegnet_ae_model(device=device)
    elif method == "pca_pc" or method == "pca_avg":
        model = FrozenPCATorch("latent_extraction/pca/models/pca_pc_psd_k8.npz", device=device)
    elif method == "psd_ae_pc" or method == "psd_ae_avg":
        model = get_psd_ae_model(device=device)
    
    for item in data:
        # Require (raw, g, a, ab, sample_id)
        if len(item) != 5:
            raise ValueError("Expected 5-tuple (raw, gender, age, abnormal, sample_id) from data loader.")
        x, g, a, ab, sample_id = item
        
        if method == "ctm_nn_pc":
            latent_feature = infer_latent_parameters(model, x, device=device, per_channel=True)
        elif method == "ctm_nn_avg":
            latent_feature = infer_latent_parameters(model, x, device=device, per_channel=False)
        elif method == "ctm_cma_pc":
            latent_feature = fit_ctm_per_channel_from_raw(x)
        elif method == "ctm_cma_avg":
            latent_feature = fit_ctm_average_from_raw(x)
        elif method == "jr_pc":
            latent_feature = fit_jr_per_channel_from_raw(x)
        elif method == "jr_avg":
            latent_feature = fit_jr_average_from_raw(x)
        elif method == "wong_wang_pc":
            latent_feature = fit_wong_wang_per_channel_from_raw(x)
        elif method == "wong_wang_avg":
            latent_feature = fit_wong_wang_average_from_raw(x)
        elif method == "hopf_pc":
            latent_feature = hopf.fit_hopf_from_raw(x, per_channel=True)
        elif method == "hopf_avg":
            latent_feature = hopf.fit_hopf_from_raw(x, per_channel=False)
        elif method == "c22":
            latent_feature = extract_c22(x)
        elif method == "pca_pc":
            latent_feature = extract_pca_from_raw(x, model=model, device=device, per_channel=True)  
        elif method == "pca_avg":
            latent_feature = extract_pca_from_raw(x, model=model, device=device, per_channel=False)  
        elif method == "psd_ae_pc":
            latent_feature = extract_psd_ae_channel(x, device=device, model=model)
        elif method == "psd_ae_avg":
            latent_feature = extract_psd_ae_avg(x, device=device, model=model)
        elif method == "eegnet":
            latent_feature = extract_eegnet_ae(x, device=device, model=model)  # type: ignore
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if latent_feature is None:
            raise ValueError(f"No latent feature extracted for {method}")
    
        if np.ndim(latent_feature) != 1:
            raise ValueError(f"Latent feature must be 1D, got {np.ndim(latent_feature)}D")

        # Ensure output is a float32 tensor for downstream .detach() usage
        latent_feature = ensure_float32_tensor(latent_feature)

        # Serialize safely
        record = make_latent_record(latent_feature, g, a, ab, sample_id)

        # Append to the file
        if save_path:
            append_jsonl(save_path, record)

        # Dataset keeps tuples (latent, g, a, ab) to avoid downstream changes
        latent_features.append((latent_feature, g, a, ab))

    return DataLoader(latent_features, batch_size=batch_size, shuffle=False)
