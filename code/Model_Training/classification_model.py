# -------------------------------------------------------------
# multitask_paramnet.py
# Estimate gender, age and abnormal / normal from a JR vector
# -------------------------------------------------------------
import numpy as np
import torch
from torch import nn


# ------------------------------------------------------------------
# 1.  Dataset helper
# ------------------------------------------------------------------
class ParamDataset(torch.utils.data.Dataset):
    """
    Generic (X, y_gender, y_age, y_abn) container.

    Parameters
    ----------
    X         : (N, D)   float32   – parameter vectors
    gender    : (N,)     int32/uint8  – 0 or 1
    age       : (N,)     float32      – years
    abnormal  : (N,)     int32/uint8  – 0 or 1
    """
    def __init__(self, X, gender, age, abnormal):
        self.X         = torch.as_tensor(X,        dtype=torch.float32)
        self.gender    = torch.as_tensor(gender,   dtype=torch.float32)
        self.age       = torch.as_tensor(age,      dtype=torch.float32)
        self.abnormal  = torch.as_tensor(abnormal, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return ( self.X[idx],
                 self.gender[idx],
                 self.age[idx],
                 self.abnormal[idx] )


# ------------------------------------------------------------------
# 2.  Multi-task network
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
                 hidden_dims=(64, 32),
                 dropout: float = 0.1):
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

    def forward(self, x):
        h = self.trunk(x)
        gender_logit = self.gender_head(h).squeeze(-1)
        age_pred     = self.age_head(h).squeeze(-1)
        abn_logit    = self.abn_head(h).squeeze(-1)

        gender = torch.sigmoid(gender_logit)   # (N,)
        abnormal = torch.sigmoid(abn_logit)    # (N,)

        return gender, age_pred, abnormal
    
    def g_loss(self, ĝ, g): return nn.BCELoss()(ĝ, g)
    def a_loss(self, â, a): return nn.MSELoss()(â, a)
    def abn_loss(self, âbn, ab): return nn.BCELoss()(âbn, ab)


# ------------------------------------------------------------------
# 3.  Training utility
# ------------------------------------------------------------------
def train(model,
          dataloader,
          n_epochs=50,
          lr=1e-3,
          device='cpu',
          λ_gender=1.0,
          λ_age=0.1,
          λ_abn=1.0):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')

    bce = nn.BCELoss()
    mse = nn.MSELoss()

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0

        for x, g, a, ab in dataloader:
            x, g, a, ab = x.to(device), g.to(device), a.to(device), ab.to(device)
            
            x = x.float()
            g  = (g == 2).float()   # map to 0/1
            a  = a.float()
            ab = ab.float()
            
            optim.zero_grad()
            ĝ, â, âbn = model(x)

            loss = (λ_gender * bce(ĝ, g) +
                    λ_abn   * bce(âbn, ab)
                    + λ_age   * mse(â, a)
                    )

            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)

        print(f"Epoch {epoch:03d}:  loss = {running / len(dataloader.dataset):.4f}")
