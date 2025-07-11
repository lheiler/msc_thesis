import torch

def extract_z(model, x, device=None):
    if 'A1' in x.ch_names:
        x.drop_channels(['A1'])
    if 'A2' in x.ch_names:
        x.drop_channels(['A2'])

    eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
                    'Cz', 'Pz', 'Fz']

    x = x.copy().pick_channels(eeg_channels)
    x = x.get_data()
    
    device = next(model.parameters()).device  # Always use model's device
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        s1 = model.stem(x)
        s2 = model.enc1(s1)
        s3 = model.enc2(s2)
        s4 = model.enc3(s3)
        s4 = model.se(s4)
        t = model.transformer(s4.permute(0, 2, 1)).permute(0, 2, 1)
        z = model.fc_enc(t.flatten(1))[0]

    return z.cpu().detach().numpy()