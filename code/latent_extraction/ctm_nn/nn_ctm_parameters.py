import pathlib
import numpy as np
import torch
from torch import Tensor
import mne
from scipy.signal import welch
# Model constants
FREQ_MIN = 1.0  # Hz
FREQ_MAX = 45.0  # Hz
N_FREQS = 513  # grid resolution
FREQS = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)


class ParameterRegressor(torch.nn.Module):
    """Simple feedforward network that maps a PSD to CTM parameters."""

    def __init__(self, in_dim: int = N_FREQS, hidden_dims=(512, 256), out_dim: int = 8):
        super().__init__()
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


def _normalise(psd: np.ndarray) -> np.ndarray:
    """Log-transform & mean-centre to capture *shape* not absolute magnitude."""
    log_psd = np.log10(psd + 1e-12)
    return log_psd - log_psd.mean()


def smooth_psd(psd: np.ndarray) -> np.ndarray:
    """Apply smoothing to PSD using convolution."""
    return np.convolve(psd, np.ones(10)/10, mode='same')


def infer_latent_parameters(model, x: mne.io.Raw, device: str = "cuda") -> np.ndarray:
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
    
    if 'A1' in x.ch_names:
        x.drop_channels(['A1'])
    if 'A2' in x.ch_names:
        x.drop_channels(['A2'])
        
    # specifically pick the 19 EEG channels by name
    eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                    'Cz', 'Pz', 'Fz']    
    
    
    model.eval()
    x = x.pick_channels(eeg_channels)
    x = x.filter(3.0, 45.0, fir_design='firwin')
    sfreq = x.info['sfreq']
    x = x.get_data()
    freqs, psds = welch(x, fs=sfreq, nperseg=(N_FREQS-1)*2)
    psds = psds.reshape(-1, psds.shape[1])
    #psds = np.array([psds.mean(axis=0)]) uncomment this to use the mean of the psds
    #print(psds.shape)
    # Process data
    all_preds = []
    
    with torch.no_grad():
        for i in range(psds.shape[0]):  
            emp_input = torch.as_tensor(
                _normalise(smooth_psd(psds[i])), 
                dtype=torch.float32
            ).to(device)
            pred = model(emp_input)[0].cpu().numpy().flatten()
            all_preds.append(pred)
    return np.array(all_preds).flatten()
