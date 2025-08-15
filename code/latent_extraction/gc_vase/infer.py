from pathlib import Path
from typing import Optional, Literal, Tuple

import numpy as np
import torch


def _load_split_model_module() -> object:
    """
    Dynamically load GC-VASE's SplitLatentModel from the vendored code without
    requiring it to be installed as a package.
    """
    split_model_path = (
        Path(__file__).resolve().parent
        / "GC-VASE"
        / "gc_vase"
        / "split_model.py"
    )
    if not split_model_path.exists():
        raise FileNotFoundError(f"GC-VASE split_model.py not found at {split_model_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gc_vase_split_model", str(split_model_path)
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_gc_vase_model(
    *,
    in_channels: int,
    device: torch.device,
    model_path: Optional[str] = None,
    channels: int = 256,
    latent_dim: int = 64,
    num_layers: int = 0,
    kernel_size: int = 4,
) -> torch.nn.Module:
    """
    Build the GC-VASE SplitLatentModel and (optionally) load weights.

    We filter the state_dict to only keys that exist and match shape so the model
    can be used even if the checkpoint was trained with a different channel
    configuration.
    """
    split_mod = _load_split_model_module()
    def build(n_layers: int) -> torch.nn.Module:
        return split_mod.SplitLatentModel(
            in_channels,
            channels,
            latent_dim,
            n_layers,
            kernel_size,
            recon_type="mse",
            content_cosine=True,
        )

    model = build(num_layers)

    if model_path is None:
        # Default to vendored checkpoint if present
        default_ckpt = (
            Path(__file__).resolve().parent / "GC-VASE" / "640-model.pt"
        )
        model_path = str(default_ckpt) if default_ckpt.exists() else None

    load_stats = None
    if model_path is not None and Path(model_path).exists():
        try:
            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=True)  # PyTorch >= 2.4
            except TypeError:
                ckpt = torch.load(model_path, map_location="cpu")  # Older PyTorch
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            # Filter by keys and shapes
            current_sd = model.state_dict()
            filtered = {
                k: v for k, v in state_dict.items()
                if k in current_sd and tuple(v.shape) == tuple(current_sd[k].shape)
            }
            incompatible = model.load_state_dict(filtered, strict=False)
            # Logging: how many loaded vs missing
            num_total = len(current_sd)
            num_loaded = len(filtered)
            num_missing = len(incompatible.missing_keys)
            num_unexpected = len(incompatible.unexpected_keys)
            load_stats = (num_loaded, num_total)
            print(
                f"[GC-VASE] Loaded checkpoint '{model_path}': loaded={num_loaded}/{num_total}, "
                f"missing={num_missing}, unexpected_in_ckpt={num_unexpected}",
                flush=True,
            )
            # Explicitly check first input conv presence
            first_conv_prefix = "encoder_in.0.proj."
            first_conv_ok = any(k.startswith(first_conv_prefix) for k in filtered.keys())
            if not first_conv_ok:
                print(
                    "[GC-VASE] WARNING: Input projection (encoder_in.0.proj.*) not loaded from checkpoint. "
                    "Accuracy may be near chance until adapted.",
                    flush=True,
                )
        except Exception:
            # Proceed with randomly initialized weights
            pass

    # Heuristic retry: if very few weights loaded and num_layers != 0, try num_layers=0 (training default)
    try:
        if load_stats is not None:
            loaded, total = load_stats
            frac = (loaded / max(1, total))
            if frac < 0.6 and num_layers != 0 and model_path is not None and Path(model_path).exists():
                print("[GC-VASE] Low load fraction detected; retrying with num_layers=0 to match training default.", flush=True)
                model = build(0)
                try:
                    try:
                        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
                    except TypeError:
                        ckpt = torch.load(model_path, map_location="cpu")
                    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                    current_sd = model.state_dict()
                    filtered = {k: v for k, v in state_dict.items() if k in current_sd and tuple(v.shape) == tuple(current_sd[k].shape)}
                    incompatible = model.load_state_dict(filtered, strict=False)
                    num_total = len(current_sd)
                    num_loaded = len(filtered)
                    num_missing = len(incompatible.missing_keys)
                    num_unexpected = len(incompatible.unexpected_keys)
                    print(
                        f"[GC-VASE] Retry loaded: loaded={num_loaded}/{num_total}, missing={num_missing}, unexpected_in_ckpt={num_unexpected}",
                        flush=True,
                    )
                except Exception:
                    pass
    except Exception:
        pass

    model.to(device)
    model.eval()
    return model


def _raw_to_windows(
    raw,
    *,
    window_seconds: float = 2.0,
    target_sfreq: float = 128.0,
    num_windows: int = 30,
    data_mean: Optional[float] = None,
    data_std: Optional[float] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Convert an mne.io.Raw object to evenly spaced windows of shape (N, C, T).

    Returns the windows tensor and the actual sampling frequency after resample.
    """
    # Work on a copy and ensure data is in memory
    x = raw.copy().load_data()

    # Resample to target frequency if needed
    sfreq = float(x.info.get("sfreq", target_sfreq))
    if abs(sfreq - target_sfreq) > 1e-3:
        x.resample(target_sfreq, npad="auto")
        sfreq = float(target_sfreq)

    data = x.get_data()  # (C, T)
    data = data.astype(np.float32, copy=False)

    # Normalization: prefer dataset-level stats if provided; else per-recording
    if data_mean is not None and data_std is not None:
        mean_val = float(data_mean)
        std_val = float(max(1e-8, data_std))
    else:
        mean_val = float(data.mean())
        std_val = float(data.std() + 1e-8)
    data = (data - mean_val) / std_val

    num_channels, total_T = data.shape
    target_T = int(round(window_seconds * sfreq))
    if total_T < target_T:
        # Pad to at least one window
        pad_T = target_T - total_T
        data = np.pad(data, ((0, 0), (0, pad_T)), mode="constant")
        total_T = data.shape[1]

    if num_windows <= 1:
        # Centered single window
        start = max(0, (total_T - target_T) // 2)
        indices = [start]
    else:
        max_start = total_T - target_T
        if max_start <= 0:
            indices = [0]
        else:
            # Evenly spaced starts across the recording
            indices = np.linspace(0, max_start, num=num_windows, dtype=int).tolist()

    windows = []
    for s in indices:
        w = data[:, s : s + target_T]
        windows.append(w)
    windows_np = np.stack(windows, axis=0)  # (N, C, T)
    windows_t = torch.from_numpy(windows_np)
    return windows_t, sfreq


@torch.inference_mode()
def extract_gc_vase(
    raw,
    *,
    device: torch.device,
    model: torch.nn.Module,
    pooling: Literal["mean", "median"] = "median",
    return_dim: Literal["latent", "flatten"] = "latent",
    single_window: Optional[Literal["middle", "first", "random"]] = None,
    single_window_index: Optional[int] = None,
    num_windows: int = 30,
    data_mean: Optional[float] = None,
    data_std: Optional[float] = None,
) -> torch.Tensor:
    """
    Extract a subject-level embedding from an mne.io.Raw using GC-VASE.

    Steps:
    - Cut the recording into 2 s windows at 128 Hz
    - Encode each window with GC-VASE subject encoder
    - Pool window embeddings into a single subject vector
    """
    windows_t, sfreq = _raw_to_windows(
        raw,
        window_seconds=2.0,
        target_sfreq=128.0,
        num_windows=num_windows,
        data_mean=data_mean,
        data_std=data_std,
    )

    # Model expects (B, C, T)
    # Ensure channels match model expectation: model was trained with 30 channels by default.
    # We pass whatever channels are present (e.g., 19 EEG channels after clean_x),
    # as the model can run with different in_channels at init time. Just move to device.
    windows_t = windows_t.to(device, non_blocking=True)

    # Subject-task split encoding; take subject embedding
    if not hasattr(model, "subject_task_encode"):
        raise AttributeError("GC-VASE model missing subject_task_encode method")

    subject_latents, _ = model.subject_task_encode(windows_t)
    # subject_latents: (N, latent_dim * latent_seqs)
    if return_dim == "latent":
        # Reshape to (N, S, D) and pool across latent sequence S -> (N, D)
        try:
            subject_latents = subject_latents.view(
                subject_latents.shape[0], model.latent_seqs, model.latent_dim
            ).mean(dim=1)
        except Exception:
            # Fallback: infer S from total size
            total = subject_latents.shape[1]
            S = getattr(model, "latent_seqs", None)
            D = getattr(model, "latent_dim", None)
            if S is None or D is None or S * D != total:
                return_dim = "flatten"
            else:
                subject_latents = subject_latents.view(-1, S, D).mean(dim=1)

    # If a single window is requested, select it and return directly (no across-window pooling)
    if single_window is not None or single_window_index is not None:
        N = subject_latents.shape[0]
        if single_window_index is not None:
            idx = int(max(0, min(N - 1, single_window_index)))
        else:
            if single_window == "first":
                idx = 0
            elif single_window == "random":
                idx = int(torch.randint(low=0, high=N, size=(1,)).item())
            else:
                idx = N // 2  # middle
        subject_vector = subject_latents[idx]
    else:
        # Pool across windows
        if pooling == "mean":
            subject_vector = subject_latents.mean(dim=0)
        else:
            subject_vector = subject_latents.median(dim=0).values

    return subject_vector.detach().float().cpu()


