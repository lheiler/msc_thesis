from Latent_Extraction.cortico_thalamic import fit_ctm_from_raw 
from torch.utils.data import DataLoader
from Latent_Extraction.c22 import extract_c22
from Latent_Extraction.c22 import extract_c22_psd

def extract_latent_features(data: DataLoader, batch_size, method, save_path=""):
    """
    Extract latent features from the EEG data using the Cortico-Thalamic Model (CTM).

    Parameters
    ----------
    data : DataLoader
        DataLoader with EEG data.
    batch_size : int
        Batch size for processing.

    Returns
    -------
    list
        List of extracted latent features.
    """
    latent_features = []
  
    for x, g, a, ab in data:    
        
        if method == "ctm": latent_feature = fit_ctm_from_raw(x)
        if method == "c22": latent_feature = extract_c22(x)
        if method == "c22_psd": latent_feature = extract_c22_psd(x)
        # save intermediate results to pipeline/Results/tuh-eeg-ctm-parameters/temp_latent_features.txt
        
        with open(save_path, "a") as f:
            f.write(f"{(latent_feature,g,a,ab)}\n")
        
        latent_features.append((latent_feature, g, a, ab))

    return DataLoader(latent_features, batch_size=batch_size, shuffle=False)
