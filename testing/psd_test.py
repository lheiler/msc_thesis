
import mne
import numpy as np
import sys
sys.path.append("/rds/general/user/lrh24/home/thesis/code")
from utils.util import compute_psd_from_raw
import matplotlib.pyplot as plt
import torch

edf_path = "/rds/general/user/lrh24/ephemeral/edf/train/normal/01_tcp_ar/aaaaakfp_s001_t000.edf"

# Load the raw EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")


raw_length = raw.times[-1]
raw.crop(tmin=raw_length - 10, tmax=raw_length)

# Compute PSD using the shared utility (returns (C, F) for per-channel)
psd, freqs = compute_psd_from_raw(raw, calculate_average=True, normalize=True, return_freqs=True)


plt.plot(freqs, psd) 
#plt.yscale("log")
plt.savefig("/rds/general/user/lrh24/home/thesis/testing/psd.png")
