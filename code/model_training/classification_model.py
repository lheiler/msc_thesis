# -------------------------------------------------------------
# multitask_paramnet.py
# Estimate gender, age and abnormal / normal from a JR vector
# -------------------------------------------------------------
import numpy as np
import torch
from torch import nn


# ------------------------------------------------------------------
# 1.  Multi-task network
# ------------------------------------------------------------------
class ClassificationModel(nn.Module):
    """
    Shared MLP → three heads:
        * gender     : sigmoid        (binary cross-entropy)
        * age        : linear         (MSE / MAE)
        * abnormal   : sigmoid        (binary cross-entropy)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims=(256, 128, 32),
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        dims = (input_dim, *hidden_dims)
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_f, out_f),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.gender_head   = nn.Linear(last_dim, 1)
        self.age_head      = nn.Linear(last_dim, 1)
        self.abn_head      = nn.Linear(last_dim, 1)
        
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

        self.age_mean = 0.0
        self.age_std = 1.0

    def forward(self, x):
        h = self.trunk(x)
        gender_logit = self.gender_head(h).squeeze(-1)
        age_pred     = self.age_head(h).squeeze(-1)
        abn_logit    = self.abn_head(h).squeeze(-1)

        gender = torch.sigmoid(gender_logit)   # (N,)
        abnormal = torch.sigmoid(abn_logit)    # (N,)

        return gender, age_pred, abnormal
    
    def g_loss(self, ĝ, g): return self.bce(ĝ, g)
    def a_loss(self, â, a): return self.mse(â, a)
    def abn_loss(self, âbn, ab): return self.bce(âbn, ab)

    def normalize_age(self, a):
        return (a - self.age_mean) / self.age_std

    def denormalize_age(self, a):
        return a * self.age_std + self.age_mean


# ------------------------------------------------------------------
# 2.  Training utility
# ------------------------------------------------------------------
def train(model,
          dataloader,
          n_epochs=50,
          lr=1e-3,
          device='cpu',
          λ_gender=0.0,
          λ_age=0.0,
          λ_abn=1.0):
    
    
    device = torch.device(device)
    # get mean and std from the dataloader for age normalization
    age_mean = 0.0
    age_std = 0.0
    for _, _, a, _ in dataloader:
        age_mean += a.sum().item()
        age_std += (a ** 2).sum().item()
    age_mean /= len(dataloader.dataset)
    age_std = (age_std / len(dataloader.dataset) - age_mean ** 2) ** 0.5
    print(f"Age mean: {age_mean:.2f}, Age std: {age_std:.2f}")
    
    model.age_mean = age_mean
    model.age_std = age_std
    
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0

        for x, g, a, ab in dataloader:
            x, g, a, ab = x.to(device), g.to(device), a.to(device), ab.to(device)

            x = x.float()
            g = (g.long() == 2).float().detach()  # Safe runtime conversion
            a = a.float().detach()
            a = model.normalize_age(a)
            ab = ab.float().detach()

            optim.zero_grad()

            ĝ, â, âbn = model(x)

            # Split losses
            loss_gender = model.g_loss(ĝ, g)
            loss_abn    = model.abn_loss(âbn, ab)
            loss_age    = model.a_loss(â, a)

            loss = λ_gender * loss_gender + λ_abn * loss_abn + λ_age * loss_age
            print("predicted age:", â[0].item(), "true age:", a[0].item())
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)

        print(f"Epoch {epoch:03d}:  loss = {running / len(dataloader.dataset):.4f}")
