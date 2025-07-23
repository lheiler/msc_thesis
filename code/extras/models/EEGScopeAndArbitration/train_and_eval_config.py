# Configuration file for train_and_eval.py

# ------------------------------------------------------------
# General training options
# ------------------------------------------------------------
# CSV file where train_and_eval.py will append one row per experiment
log_path = "result.csv"

# Plot learning curves (set to True if you want additional PNGs written)
plot_result = False

# Early-stopping settings
earlystopping = True
es_patience = 10

# When True, a new model is trained; set to False to load an existing
train_whole_dataset_again = True
# If you have previously trained models and only wish to evaluate them,
# set test_model = True
test_model = False

# ------------------------------------------------------------
# MNE / logging
# ------------------------------------------------------------
# Valid values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
# WARNING keeps the console reasonably quiet while still surfacing issues.
mne_log_level = "WARNING"

# ------------------------------------------------------------
# Model-specific hyper-parameters (copied from the template so that
# they are easy to tweak later). You can safely ignore these unless you
# want to try a different architecture.
# ------------------------------------------------------------

# Deep4Net
deep4_n_filters_time = 25
deep4_n_filters_spat = 25
deep4_filter_time_length = 10
deep4_pool_time_length = 3
deep4_pool_time_stride = 3
deep4_n_filters_2 = 50
deep4_filter_length_2 = 10
deep4_n_filters_3 = 100
deep4_filter_length_3 = 10
deep4_n_filters_4 = 200
deep4_filter_length_4 = 10
deep4_first_pool_mode = "max"
deep4_later_pool_mode = "max"
deep4_double_time_convs = False
deep4_split_first_layer = True
deep4_batch_norm = True
deep4_batch_norm_alpha = 0.1
deep4_stride_before_pool = False

# ShallowFBCSPNet
shallow_n_filters_time = 40
shallow_filter_time_length = 25
shallow_n_filters_spat = 40
shallow_pool_time_length = 75
shallow_pool_time_stride = 15
shallow_split_first_layer = True
shallow_batch_norm = True
shallow_batch_norm_alpha = 0.1

# TCN-1
tcn_kernel_size = 11
tcn_n_blocks = 5
tcn_n_filters = 55
tcn_add_log_softmax = True
tcn_last_layer_type = "max_pool"

# Vision Transformer
vit_patch_size = 10
vit_dim = 64
vit_depth = 6
vit_heads = 16
vit_mlp_dim = 64
vit_emb_dropout = 0.1 