# --------------------------------------------------
#  Latent feature extraction methods
# --------------------------------------------------
from latent_extraction.cortico_thalamic import fit_ctm_from_raw
from torch.utils.data import DataLoader
from latent_extraction.c22 import extract_c22, extract_c22_psd
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
# For CwA-T we will lazily import to avoid unnecessary dependency cost when
# other methods are used.  Import inside functions below.
import os

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
    elif method in {"conv_ae"}:

        
        # # Optional latent dim from env, else default 128
        # latent_dim = int(os.environ.get("CONVAE_LATENT_DIM", "128"))
        # model = conv_mod.get_conv_ae_model(
        #     device=device,
        #     model_path=conv_model_path,
        #     n_channels=19,
        #     input_window_samples=60 * 128,
        #     latent_dim=latent_dim,
        # )
        extract_conv_ae = conv_mod.extract_conv_ae  # type: ignore
    elif method in {"pca_psd"}:
        # Lazy-load PCA runtime and prepare PSD parameters
        from latent_extraction.pca.pca import FrozenPCATorch  # type: ignore
        pca_model_path = "/homes/lrh24/thesis/code/latent_extraction/pca/models/pca_avg_psd_k8.npz"
        try:
            model = FrozenPCATorch(pca_model_path, device=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PCA artifact at '{pca_model_path}'. "
                f"Set PCA_MODEL_PATH or generate with latent_extraction/pca/pca.py. Error: {e}"
            )

        # PSD params (keep aligned with fitter defaults; overridable via env)
        psd_fmin = 1.0
        psd_fmax = 45.0
        psd_n_per_seg = 512
        psd_n_fft = 512
        psd_log = True
        psd_resample = None
    elif method in {"psd_ae", "psd_ae_avg", "psd_ae_channel", "psd_ae_ch"}:
        # Load PSD-AE model and extraction helpers
        from latent_extraction.psd_ae.psd_ae import (
            get_psd_ae_model,
            extract_psd_ae_avg as _extract_psd_ae_avg,
            extract_psd_ae_channel as _extract_psd_ae_channel,
        )  # type: ignore
        model = get_psd_ae_model(device=device)
    elif method in {"gc_vase", "gcvase", "gc-vase"}:
        # Lazy import GC-VASE infer helpers
        this_dir = Path(__file__).resolve().parent
        infer_path = this_dir / "gc_vase" / "infer.py"
        spec = importlib.util.spec_from_file_location("gc_vase_infer", str(infer_path))
        assert spec and spec.loader, f"Could not load spec for {infer_path}"
        gcv_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gcv_mod)
        # Build/load model once
        in_channels = 19  # we pick EEG channels earlier via clean_x
        # Resolve model checkpoint strictly: pick newest gcvase_*.pt; no fallbacks
        base_dir = this_dir / "gc_vase" / "GC-VASE"
        gcv_model_path = None
        candidates = sorted(base_dir.glob("gcvase_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            gcv_model_path = str(candidates[0])
        if gcv_model_path is None:
            raise RuntimeError(
                f"No GC-VASE checkpoint found in {base_dir}. Place a file named gcvase_*.pt with encoded hyperparameters."
            )
        # Parse hyperparameters from filename (required)
        import re
        fname = os.path.basename(gcv_model_path)
        m = re.search(r"_c(\d+)_d(\d+)_L(\d+)_k(\d+)_", fname)
        if not m:
            raise RuntimeError(
                "GC-VASE checkpoint filename must include _c{channels}_d{latent}_L{layers}_k{kernel}_ pattern"
            )
        gcv_channels = int(m.group(1))
        gcv_latent_dim = int(m.group(2))
        gcv_num_layers = int(m.group(3))
        gcv_kernel_size = int(m.group(4))
        model = gcv_mod.get_gc_vase_model(
            in_channels=in_channels,
            device=device,
            model_path=gcv_model_path,
            channels=gcv_channels,
            latent_dim=gcv_latent_dim,
            num_layers=gcv_num_layers,
            kernel_size=gcv_kernel_size,
        )
        extract_gc_vase = gcv_mod.extract_gc_vase  # type: ignore
        # Load dataset stats strictly from generic_data.pt next to the model; no fallbacks
        model_dir = Path(gcv_model_path).parent
        data_stats_path = model_dir / "generic_data.pt"
        if not data_stats_path.exists():
            raise RuntimeError(f"Dataset stats not found: {data_stats_path}")
        _ds = torch.load(str(data_stats_path), map_location="cpu")
        if not (isinstance(_ds, dict) and "data_mean" in _ds and "data_std" in _ds):
            raise RuntimeError(f"Invalid dataset stats file: {data_stats_path}")
        dm = _ds["data_mean"]
        ds = _ds["data_std"]
        gcv_data_mean = float(dm.detach().cpu().item() if hasattr(dm, "detach") else dm)
        gcv_data_std = float(ds.detach().cpu().item() if hasattr(ds, "detach") else ds)
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
        elif method in {"eegnet"}:
            # Encoder output vector
            latent_feature = extract_eegnet_ae(x, device=device, model=model)  # type: ignore
        elif method in {"conv_ae", "convae", "ConvAE", "convAE"}:
            latent_feature = extract_conv_ae(x, device=device, model=model)  # type: ignore
        elif method in {"pca"}:
            psd = x.compute_psd(method="welch",fmin=psd_fmin,fmax=psd_fmax,n_per_seg=psd_n_per_seg, n_fft=psd_n_fft, verbose="ERROR")
            psds, freqs = psd.get_data(return_freqs=True)
            avg_psd = psds.mean(axis=0).astype(np.float32)
            if psd_log: avg_psd = np.log10(np.maximum(avg_psd, 1e-12))
            latent_feature = model.transform_vec(avg_psd)  # type: ignore
        elif method in {"hopf"}:
            latent_feature = fit_hopf_from_raw(x)
        elif method in {"psd_ae", "psd_ae_avg"}:
            latent_feature = _extract_psd_ae_avg(x, device=device, model=model)
        elif method in {"psd_ae_channel", "psd_ae_ch"}:
            latent_feature = _extract_psd_ae_channel(x, device=device, model=model)
        elif method in {"gc_vase", "gcvase", "gc-vase"}:
            # GC-VASE subject-level vector: single 2s middle window (64-D latent)
            latent_feature = extract_gc_vase(
                x,
                device=device,
                model=model,
                pooling="median",
                return_dim="latent",
                single_window="middle",
                data_mean=gcv_data_mean,
                data_std=gcv_data_std,
            )  # type: ignore
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
