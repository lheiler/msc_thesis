import torch

def extract_z(model, x, device="cpu"):
    if 'A1' in x.ch_names:
        x.drop_channels(['A1'])
    if 'A2' in x.ch_names:
        x.drop_channels(['A2'])
        
    # specifically pick the 19 EEG channels by name
    eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                    'Cz', 'Pz', 'Fz']    
    
    x = x.copy().pick_channels(eeg_channels)
    x = x.get_data()
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, 19, time]
    s3 = model._encode(x)[2]
    z = s3.flatten(1)     # Shape: [B, 128 * 960] = [B, 122880]
    z = model.fc_enc(z)[0]
    # get a copy of z on CPU
    return z.cpu().detach().numpy()  # Shape: [B, 128
