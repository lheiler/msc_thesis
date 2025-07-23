"""
Hyper-parameter grid for train_and_eval.py

Based on batch_test_hyperparameters.default.py with paths adapted to the
local machine.  The TUH Abnormal EEG dataset is assumed to live under
/vol/bitbucket/lrh24/edf with the usual `train/` and `eval/` sub-folders.
TUEG is disabled for now; enable it later by setting `TUEG=[True]` and
providing a valid `TUEG_PATH`.
"""

# ---------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------
MNE_LOG_LEVEL        = ["WARNING"]     # keep console noise low

# ---------------------------------------------------------------------
# Dataset selection & preprocessing
# ---------------------------------------------------------------------
REMOVE_ATTRIBUTE     = [None]          # do not drop by patients/sessions
N_REPETITIONS        = 1
RANDOM_STATE         = [87]

TUAB                 = [True]          # use TUAB abnormal dataset
TUEG                 = [False]         # do not use TUEG for now

# How many recordings to load (None -> all available)
N_TUAB               = [None]
N_TUEG               = [None]
N_LOAD               = [100]           # only used when LOAD_SAVED_* is True
PRELOAD              = [True]
WINDOW_LEN_S         = [60]            # 1-min windows

# Paths to raw datasets
TUAB_PATH            = ["/vol/bitbucket/lrh24/edf"]
TUEG_PATH            = ["/vol/bitbucket/lrh24/edf"]  # placeholder

# Whether to save/restore intermediate pre-processed files
SAVED_DATA           = [False]
SAVED_PATH           = ["/vol/bitbucket/lrh24/eeg_saved_data"]
SAVED_WINDOWS_DATA   = [False]
SAVED_WINDOWS_PATH   = ["/vol/bitbucket/lrh24/eeg_saved_windows"]
LOAD_SAVED_DATA      = [False]
LOAD_SAVED_WINDOWS   = [False]

# Filtering / standardization
BANDPASS_FILTER      = [False]
LOW_CUT_HZ           = [4.]
HIGH_CUT_HZ          = [38.]

STANDARDIZATION      = [True]
FACTOR_NEW           = [1e-3]
INIT_BLOCK_SIZE      = [1000]

# ---------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------
N_JOBS               = [8]
N_CLASSES            = [2]
LR                   = [0.001]
WEIGHT_DECAY         = [0.0005]
BATCH_SIZE           = [1]
N_EPOCHS             = [30]

# Recording trimming
TMIN                 = [5 * 60]        # skip first 5 min
TMAX                 = [None]          # no upper bound
MULTIPLE             = [0]
SEC_TO_CUT           = [60]
DURATION_RECORDING_SEC = [20 * 60]
MAX_ABS_VAL          = [800]
SAMPLING_FREQ        = [100]

# Train/val/test split
TEST_ON_VAL          = [True]
SPLIT_WAY            = ["train_on_tuab_tueg_test_on_tueg"]
TRAIN_SIZE           = [0.8]
VALID_SIZE           = [0.1]
TEST_SIZE            = [0.1]
SHUFFLE              = [True]

# Model
MODEL_NAME           = ["deep4"]
DEEP4_BATCH_NORM_ALPHA = [0.1]
FINAL_CONV_LENGTH    = ["auto"]
DROPOUT              = [0.1]
WINDOW_STRIDE_SAMPLES= [None]

# Relabeling (not used)
RELABEL_DATASET      = [[]]
RELABEL_LABEL        = [[]]

# EEG channels to retain (unchanged)
CHANNELS = [[
    "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF",
    "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", "EEG O1-REF", "EEG O2-REF",
    "EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF", "EEG T5-REF",
    "EEG T6-REF", "EEG FZ-REF", "EEG PZ-REF", "EEG CZ-REF",
]]

# Activation function(s) to test
ACTIVATION           = ["elu"] 