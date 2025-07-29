import mne

eeg_path = "/Users/lorenzheiler/small_dataset/eval/abnormal/01_tcp_ar/aaaaaddm_s006_t000.edf"
eeg_path2 = "/Users/lorenzheiler/small_dataset/eval/abnormal/01_tcp_ar/aaaaakjk_s002_t002.edf"


raw = mne.io.read_raw_edf(eeg_path, preload=True)

print(raw.info)

raw2 = mne.io.read_raw_edf(eeg_path2, preload=True)


sex_code = raw.info['subject_info']['sex']
print(sex_code)