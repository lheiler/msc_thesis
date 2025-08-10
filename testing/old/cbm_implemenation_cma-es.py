# This code implements the cortico-thalamic brain model and optimizes its parameters to fit



import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
import cma
from scipy.interpolate import interp1d

# --- constants ----------------------------------------------------------
Lx = Ly = 0.5          # metres
k0 = 10.0              # m^-1
# fixed physiological constants from Table 2
gamma_e = 116.0        # s^-1
r_e     = 0.086        # m  (86 mm)

# --- frequency grid -----------------------------------------------------
f = np.arange(0.5, 45.25, 0.25)       # Hz
w = 2 * np.pi * f                     # rad/s

# --- spatial grid -------------------------------------------------------
M = 10
m = n = np.arange(-M, M+1)
kx = 2*np.pi*m[:,None]/Lx             # shape (2M+1,1)
ky = 2*np.pi*n[None,:]/Ly             # shape (1,2M+1)
k2 = kx**2 + ky**2                    # broadcast to (2M+1,2M+1)

# --- model parameters to play with (pick reasonable starts) -------------
params = dict(G_ee   = 10.3,
G_ei   = -11.2,
G_ese  = 1.7,
G_esre = -2.7,
G_srs  = -0.13,
alpha  = 58,
beta   = 305,
t0     = 0.08)
# pack any optimiser you like around this --------------------------------

def L_matrix(omega, alpha, beta):
    return 1/((1-1j*omega/alpha)*(1-1j*omega/beta))

def q2_re2(omega, p):
    Lw = L_matrix(omega, p['alpha'], p['beta'])
    num = (1 - 1j*omega/gamma_e)**2 - 1
    den = 1 - p['G_ei']*Lw
    bracket = (Lw*p['G_ee']
               + (Lw**2 * p['G_ese'] + Lw**3 * p['G_esre'])
                 * np.exp(1j*omega*p['t0'])
                 /(1 - Lw**2*p['G_srs']))
    return (num - bracket/den).real      # already multiplied by r_e^2 later

def phi_e(k2, omega, p):
    Lw = L_matrix(omega, p['alpha'], p['beta'])
    q2 = q2_re2(omega, p)
    denom = (1 - p['G_srs']*Lw**2)*(1 - p['G_ei']*Lw)*(k2*r_e**2 + q2*r_e**2)
    return p['G_ese']*np.exp(1j*omega*p['t0']/2) / denom

def P_omega(p):
    """Return model power spectrum in arbitrary units."""
    P = np.zeros_like(w, dtype=float)
    for idx, omega in enumerate(w):
        Lw = L_matrix(omega, p['alpha'], p['beta'])
        q2 = q2_re2(omega, p)
        denom = ((1 - p['G_srs']*Lw**2)
                 * (1 - p['G_ei']*Lw)
                 * (k2*r_e**2 + q2*r_e**2))
        phi = p['G_ese']*np.exp(1j*omega*p['t0']/2)/denom
        Fk  = np.exp(-k2 / k0**2)
        P[idx] = np.sum(np.abs(phi)**2 * Fk)
    Δk = (2*np.pi/Lx)*(2*np.pi/Ly)
    return P * Δk



# --- load in reference eeg data --------------------------------
subject = 1
runs = [7]  # Resting state
raw_fnames = eegbci.load_data(subject, runs)
raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)


print(raw.ch_names)
#pick FP1 channel
raw.pick_channels(['O1..'])
raw.filter(l_freq=1.0, h_freq=40.0)

# #compute power spectrum
# raw.plot_psd(fmin=1, fmax=40, average=True, show=True)
# plt.show()


# --- optimise to fit the EEG data -------------------------------

spectrum = raw.compute_psd(method='welch', fmin=1, fmax=40, n_fft=512)
psds = spectrum.get_data()
freqs = spectrum.freqs
mean_psd = psds.mean(axis=0)


def loss_function(theta, freqs, real_psd):
    keys = ['G_ee', 'G_ei', 'G_ese', 'G_esre', 'G_srs', 'alpha', 'beta', 't0']
    p = dict(zip(keys, theta))
    model_psd = P_omega(p)
    interp_func = interp1d(f, model_psd, kind='linear', bounds_error=False, fill_value="extrapolate")
    model_resampled = interp_func(freqs)

    log_model = np.log10(model_resampled + 1e-10)
    log_real = np.log10(real_psd + 1e-10)

    # Weighting for alpha range
    weights = np.ones_like(freqs)
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    weights[alpha_mask] *= 5  # emphasize alpha region

    return np.mean(weights * (log_model - log_real)**2)

theta0 = [10.3, -11.2, 1.7, -2.7, -0.13, 58, 305, 0.08]

# Standard deviations for initial sampling
sigma0 = 0.5

# Bounds for each parameter
bounds = [
    (0, 30),    # G_ee
    (-30, 0),   # G_ei
    (0, 10),    # G_ese
    (-10, 0),   # G_esre
    (-1, 0),    # G_srs
    (10, 100),  # alpha
    (100, 400), # beta
    (0.01, 0.2) # t0
]

# Extract lower and upper bounds
lower_bounds, upper_bounds = zip(*bounds)

# Run CMA-ES optimization
es = cma.CMAEvolutionStrategy(theta0, sigma0, {'bounds': [lower_bounds, upper_bounds]})
es.optimize(lambda x: loss_function(x, freqs, mean_psd))
best_theta = es.result.xbest




# --- results ---------------------------------------------------
print("Optimized parameters:")
for key, value in zip(['G_ee', 'G_ei', 'G_ese', 'G_esre', 'G_srs', 'alpha', 'beta', 't0'], best_theta):
    print(f"{key}: {value:.4f}")
print("Optimized loss function value:", es.result.fbest)

# --- plot results ----------------------------------------------
keys = ['G_ee', 'G_ei', 'G_ese', 'G_esre', 'G_srs', 'alpha', 'beta', 't0']
best_params = dict(zip(keys, best_theta))
model_fit = P_omega(best_params)

# Plot real vs. modeled power spectra

# Resample model spectrum to EEG frequency grid
interp_func = interp1d(f, model_fit, kind='linear', bounds_error=False, fill_value="extrapolate")
model_fit_resampled = interp_func(freqs)

# Convert to dB
mean_psd_db = 10 * np.log10(mean_psd + 1e-20)
model_fit_db = 10 * np.log10(model_fit_resampled + 1e-20)

plt.figure(figsize=(10, 5))
plt.plot(freqs, mean_psd_db, color='C0', label='EEG O1 (Welch)', linewidth=2)
plt.plot(freqs, model_fit_db, color='C1', label='Model fit', linestyle='--', linewidth=2)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB)")
plt.title("EEG PSD CMA-ES")
plt.grid(True, which="both", linestyle='-', linewidth=0.5)
plt.legend(loc='best')
plt.xlim(1, 40)
plt.tight_layout()
plt.savefig("results/cbm_eeg_psd_cmaes.png")