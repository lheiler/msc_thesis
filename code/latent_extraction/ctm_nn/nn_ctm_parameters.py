import pathlib
import numpy as np
import torch
from torch import Tensor
import mne
from scipy.signal import welch


from utils.util import normalize_psd, PSD_CALCULATION_PARAMS, compute_psd_from_raw

class ParameterRegressor(torch.nn.Module):
    """Simple feedforward network that maps a PSD to CTM parameters."""

    def __init__(self, in_dim: int | None = None, hidden_dims=(512, 256), out_dim: int = 8):
        super().__init__()
        if in_dim is None:
            in_dim = int(PSD_CALCULATION_PARAMS.get("n_fft", 256)) // 2 + 1
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)




def smooth_psd(psd: np.ndarray) -> np.ndarray:
    """Apply smoothing to PSD using convolution."""
    return np.convolve(psd, np.ones(10)/10, mode='same')


def infer_latent_parameters(model, raw: mne.io.Raw, device: str = "cuda", per_channel: bool = False) -> np.ndarray:
    """
    Load the trained model and perform inference to get latent parameters.
    
    Parameters
    ----------
    model_path : pathlib.Path
        Path to the saved model checkpoint (.pt file)
    data : np.ndarray
        Input PSD data with shape (N, N_FREQS) where N is the number of samples
    device : str, optional
        Device to run inference on ("cuda" or "cpu"), default "cuda"
    
    Returns
    -------
    np.ndarray
        Predicted latent parameters with shape (N, PARAM_DIM)
    """
    model.eval()
    all_preds = []
    psds = compute_psd_from_raw(raw, calculate_average=not per_channel, normalize=True)
    #compute psd for each channel gives back (C, F) with average we get (F,)
    
    
    with torch.no_grad():
        emp_input = torch.as_tensor(
            psds, 
            dtype=torch.float32
        ).to(device)
        pred = model(emp_input).cpu().numpy()
        all_preds.append(pred)
    return np.array(all_preds).flatten()
