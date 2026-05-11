"""Unified latent feature extraction dispatcher.

Routes a method name to the corresponding extraction function, handles
parallel execution for CPU-bound mechanistic models, and persists results
as JSONL for caching.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import latent_extraction.hopf as hopf
from latent_extraction.c22 import extract_c22
from latent_extraction.cortico_thalamic import (
    fit_ctm_average_from_raw,
    fit_ctm_per_channel_from_raw,
)
from latent_extraction.ctm_nn.nn_ctm_parameters import (
    ParameterRegressor,
    infer_latent_parameters,
)
from latent_extraction.EEGNet_AE.infer import extract_eegnet_ae, get_eegnet_ae_model
from latent_extraction.jansen_rit import (
    fit_jr_average_from_raw,
    fit_jr_per_channel_from_raw,
)
from latent_extraction.pca.pca import FrozenPCATorch, extract_pca_from_raw
from latent_extraction.psd_ae.psd_ae import (
    extract_psd_ae_avg,
    extract_psd_ae_channel,
    get_psd_ae_model,
)
from latent_extraction.wong_wang import (
    fit_wong_wang_average_from_raw,
    fit_wong_wang_per_channel_from_raw,
)
from utils.util import (
    append_jsonl,
    ensure_float32_tensor,
    make_latent_record,
    select_device,
    truncate_file,
)

logger = logging.getLogger(__name__)

PARALLELIZABLE_METHODS: set[str] = {
    "ctm_cma_pc", "ctm_cma_avg",
    "jr_pc", "jr_avg",
    "wong_wang_pc", "wong_wang_avg",
    "hopf_pc", "hopf_avg", "c22",
}

_METHOD_DISPATCH: dict[str, Any] = {
    "ctm_cma_pc": lambda x: fit_ctm_per_channel_from_raw(x),
    "ctm_cma_avg": lambda x: fit_ctm_average_from_raw(x),
    "jr_pc": lambda x: fit_jr_per_channel_from_raw(x),
    "jr_avg": lambda x: fit_jr_average_from_raw(x),
    "wong_wang_pc": lambda x: fit_wong_wang_per_channel_from_raw(x),
    "wong_wang_avg": lambda x: fit_wong_wang_average_from_raw(x),
    "hopf_pc": lambda x: hopf.fit_hopf_from_raw(x, per_channel=True),
    "hopf_avg": lambda x: hopf.fit_hopf_from_raw(x, per_channel=False),
    "c22": lambda x: extract_c22(x),
}


def _run_single_parallel(method: str, x: Any) -> np.ndarray:
    """Execute a single parallelizable extraction.

    Args:
        method: Extraction method name.
        x: MNE Raw object.

    Returns:
        1-D feature array.

    Raises:
        ValueError: If *method* is not in ``PARALLELIZABLE_METHODS``.
    """
    if method not in _METHOD_DISPATCH:
        raise ValueError(f"Method not supported for parallel execution: {method}")
    return _METHOD_DISPATCH[method](x)


def _attach_sample_ids(loader: DataLoader, sample_ids: list[str]) -> DataLoader:
    """Attach sample IDs to a DataLoader for downstream alignment."""
    loader.sample_ids = sample_ids  # type: ignore[attr-defined]
    return loader


def extract_latent_features(
    data: DataLoader,
    batch_size: int,
    method: str,
    save_path: str = "",
    n_workers: int = 64,
    dataset_name: str = "tuh",
) -> DataLoader:
    """Extract latent features from EEG data and optionally persist to JSONL.

    Args:
        data: DataLoader yielding ``(raw, gender, age, abnormal, sample_id)``
            tuples.
        batch_size: Batch size for the returned DataLoader.
        method: Extraction method identifier (e.g. ``"ctm_cma_avg"``).
        save_path: If non-empty, write JSONL records here.
        n_workers: Max parallel workers for CPU-bound methods.
        dataset_name: Dataset identifier (for model loading).

    Returns:
        DataLoader of ``(latent_vec, gender, age, abnormal)`` tuples with
        ``.sample_ids`` attached.

    Raises:
        ValueError: On unknown method or invalid feature dimensionality.
    """
    latent_features: list[tuple[torch.Tensor, ...]] = []
    sample_ids: list[str] = []
    model = None
    if save_path:
        truncate_file(save_path)
    device = select_device()

    if n_workers and n_workers > 1 and method in PARALLELIZABLE_METHODS:
        items = list(data)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_run_single_parallel, method, item[0]) for item in items]
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
                sample_ids.append(str(sample_id))
        return _attach_sample_ids(
            DataLoader(latent_features, batch_size=batch_size, shuffle=False),
            sample_ids,
        )

    if method in ("ctm_nn_pc", "ctm_nn_avg"):
        model = ParameterRegressor().to(device)
        state = torch.load(
            "latent_extraction/ctm_nn/amore/models/regressor.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state["model_state"])
    elif method == "eegnet":
        model = get_eegnet_ae_model(device=device, dataset_name=dataset_name)
    elif method in ("pca_pc", "pca_avg"):
        model = FrozenPCATorch(
            f"latent_extraction/pca/models/{dataset_name}_pca_pc_psd_k8.npz",
            device=device,
        )
    elif method in ("psd_ae_pc", "psd_ae_avg"):
        model = get_psd_ae_model(device=device, dataset_name=dataset_name)

    for item in data:
        if len(item) != 5:
            raise ValueError(
                "Expected 5-tuple (raw, gender, age, abnormal, sample_id) from data loader."
            )
        x, g, a, ab, sample_id = item

        if method == "ctm_nn_pc":
            latent_feature = infer_latent_parameters(model, x, device=device, per_channel=True)
        elif method == "ctm_nn_avg":
            latent_feature = infer_latent_parameters(model, x, device=device, per_channel=False)
        elif method in _METHOD_DISPATCH:
            latent_feature = _METHOD_DISPATCH[method](x)
        elif method == "pca_pc":
            latent_feature = extract_pca_from_raw(x, model=model, device=device, per_channel=True)
        elif method == "pca_avg":
            latent_feature = extract_pca_from_raw(x, model=model, device=device, per_channel=False)
        elif method == "psd_ae_pc":
            latent_feature = extract_psd_ae_channel(x, device=device, model=model)
        elif method == "psd_ae_avg":
            latent_feature = extract_psd_ae_avg(x, device=device, model=model)
        elif method == "eegnet":
            latent_feature = extract_eegnet_ae(x, device=device, model=model)
        else:
            raise ValueError(f"Unknown method: {method}")

        if latent_feature is None:
            raise ValueError(f"No latent feature extracted for {method}")
        if np.ndim(latent_feature) != 1:
            raise ValueError(f"Latent feature must be 1D, got {np.ndim(latent_feature)}D")

        latent_feature = ensure_float32_tensor(latent_feature)
        record = make_latent_record(latent_feature, g, a, ab, sample_id)
        if save_path:
            append_jsonl(save_path, record)
        latent_features.append((latent_feature, g, a, ab))
        sample_ids.append(str(sample_id))

    return _attach_sample_ids(
        DataLoader(latent_features, batch_size=batch_size, shuffle=False),
        sample_ids,
    )
