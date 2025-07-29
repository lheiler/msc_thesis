#jansen_rit test implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

import mne
from mne.datasets import eegbci

# -----------------------------------------------------------------------------
# Jansen–Rit model implementation
# -----------------------------------------------------------------------------

def _sigmoid(v: float, v0: float = 6.0, e0: float = 2.5, r: float = 0.56):
    """Sigmoidal firing response used by the Jansen–Rit model."""
    return 2.0 * e0 / (1 + np.exp(r * (v0 - v)))


def jr_rhs(t, y, params):
    """Right‑hand side (derivatives) of the Jansen–Rit ODE system.

    State vector ``y`` (6 variables):
        y0 : membrane potential of pyramidal population
        y1 : membrane potential of excitatory interneurons
        y2 : membrane potential of inhibitory interneurons
        y3‑y5 : corresponding derivatives (first‑order formulation)
    """
    A, a, B, b, C1, C2, C3, C4, P = params

    y0, y1, y2, y3, y4, y5 = y

    S_y1_y2 = _sigmoid(y1 - y2)
    S_C1y0  = _sigmoid(C1 * y0)
    S_C3y0  = _sigmoid(C3 * y0)

    dy0 = y3
    dy3 = A * a * S_y1_y2               - 2 * a * y3 - (a**2) * y0

    dy1 = y4
    dy4 = A * a * (P + C2 * S_C1y0)     - 2 * a * y4 - (a**2) * y1

    dy2 = y5
    dy5 = B * b * C4 * S_C3y0           - 2 * b * y5 - (b**2) * y2

    return [dy0, dy1, dy2, dy3, dy4, dy5]


def jr_simulate(t: np.ndarray, params, y0=None):
    """Simulate the Jansen–Rit model for time vector *t* with parameters.

    Returns the pyramidal‑population membrane potential (y1 − y2).
    """
    if y0 is None:
        y0 = np.zeros(6)  # start from rest

    sol = solve_ivp(jr_rhs,
                    t_span=(t[0], t[-1]),
                    y0=y0,
                    t_eval=t,
                    args=(params,),
                    method="RK45",
                    rtol=1e-6,
                    atol=1e-9)

    # Pyramidal output: difference between excitatory and inhibitory inputs
    return sol.y[1] - sol.y[2]

# -----------------------------------------------------------------------------
# Helper: default parameter vector
# -----------------------------------------------------------------------------

def default_params(P=120.0):
    """Return the classic Jansen–Rit parameter set with external drive *P*."""
    A = 3.25     # average synaptic gain (excitatory) [mV]
    B = 22.0     # average synaptic gain (inhibitory) [mV]
    a = 100.0    # lumped rate constant (excitatory) [s⁻¹]
    b = 50.0     # lumped rate constant (inhibitory) [s⁻¹]
    C = 135.0    # connectivity scaling
    C1 = C
    C2 = 0.8 * C
    C3 = 0.25 * C
    C4 = 0.25 * C
    return [A, a, B, b, C1, C2, C3, C4, P]

# -----------------------------------------------------------------------------
# Model fitting
# -----------------------------------------------------------------------------

def fit_external_drive(eeg: np.ndarray, t: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit only the external input *P* (Hz) so the model output ≈ EEG."""

    eeg = (eeg - np.mean(eeg)) / np.std(eeg)  # z‑score for scale‑free fit

    def objective(theta):
        P = theta[0]
        params = default_params(P=P)
        sim = jr_simulate(t, params)
        sim = (sim - np.mean(sim)) / np.std(sim)
        return np.mean((sim - eeg) ** 2)

    bounds = [(30.0, 150.0)]  # physiologically plausible range
    result = differential_evolution(objective, bounds, tol=1e-4, polish=True)
    P_opt = float(result.x[0])
    sim_opt = jr_simulate(t, default_params(P=P_opt))
    return P_opt, sim_opt

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    # ------------------------
    # 1. Load & prepare EEG
    # ------------------------
    raw_fnames = eegbci.load_data(1, runs=[7])  # subject 1, run 7 (eyes closed)
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)

    raw.pick_types(eeg=True)
    raw.filter(l_freq=1.0, h_freq=45.0, fir_design="firwin", verbose=False)
    raw.resample(128.0, verbose=False)  # speed‑up & match typical NMM fs

    channel = "Fp1."  # change me if you like!
    data, times = raw.get_data(picks=channel, return_times=True)

    # Use a 30‑second excerpt centered in the recording
    fs = raw.info["sfreq"]
    n_samp = int(10 * fs)
    start = (data.shape[1] - n_samp) // 2
    eeg = data[0, start:start + n_samp]
    t = times[start:start + n_samp] - times[start]  # start at 0 s

    # ------------------------
    # 2. Fit Jansen–Rit model
    # ------------------------
    P_opt, sim = fit_external_drive(eeg, t)
    print(f"Optimal external input P ≈ {P_opt:.2f} Hz")

    # Normalise for joint plotting
    eeg_z = (eeg - np.mean(eeg)) / np.std(eeg)
    sim_z = (sim - np.mean(sim)) / np.std(sim)

    # ------------------------
    # 3. Visualise
    # ------------------------
    plt.figure(figsize=(11, 4))
    plt.plot(t, eeg_z, label="EEG (real)", linewidth=1)
    plt.plot(t, sim_z, label="Jansen–Rit (simulated)", linewidth=1, alpha=0.8)
    plt.xlabel("Time [s]")
    plt.ylabel("Z‑scored amplitude")
    plt.title(f"Channel {channel} • subject 1, run 7")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
