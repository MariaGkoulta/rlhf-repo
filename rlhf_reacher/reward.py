import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

class RewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, dropout_prob=0.1, hidden=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 64), nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)

    def predict_reward(self, obs, action):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.forward(s, a).item()
        
def train_reward_model_batched(
    rm, pref_dataset, batch_size=64, epochs=20,
    val_frac=0.1, patience=10, optimizer=None, optimizer_lr=None, optimizer_wd=None, device='cpu', regularization_weight=1e-4,
    logger=None, iteration=0
):
    rm.to(device)
    total = len(pref_dataset)
    val_size = int(total * val_frac)
    train_size = total - val_size
    train_ds, val_ds = random_split(pref_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    no_improve = 0

    if optimizer is None:
        optimizer = torch.optim.Adam(rm.parameters(), lr=optimizer_lr, weight_decay=optimizer_wd)

    for epoch in range(1, epochs+1):
        iteration += 1
        rm.train()
        train_losses, train_accs = [], []
        for s1, a1, s2, a2, prefs in train_loader:
            N, T, obs_dim = s1.shape
            _, _, act_dim = a1.shape

            s1f = s1.view(N*T, obs_dim)
            a1f = a1.view(N*T, act_dim)
            s2f = s2.view(N*T, obs_dim)
            a2f = a2.view(N*T, act_dim)

            r1 = rm(s1f, a1f).view(N, T).sum(1)
            r2 = rm(s2f, a2f).view(N, T).sum(1)
            logits = r1 - r2

            loss = F.binary_cross_entropy_with_logits(logits, prefs)
            loss += regularization_weight * (r1.pow(2).mean() + r2.pow(2).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(((logits>0).float()==prefs).float().mean().item())

        avg_train_loss = float(np.mean(train_losses))
        avg_train_acc = float(np.mean(train_accs))
        print(f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | train_acc={avg_train_acc:.4f}", end=" | ")

        rm.train()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for s1, a1, s2, a2, prefs in val_loader:
                N, T, obs_dim = s1.shape
                s1f = s1.view(N*T, obs_dim)
                a1f = a1.view(N*T, a1.shape[2])
                s2f = s2.view(N*T, obs_dim)
                a2f = a2.view(N*T, a2.shape[2])

                r1 = rm(s1f, a1f).view(N, T).sum(1)
                r2 = rm(s2f, a2f).view(N, T).sum(1)
                logits = r1 - r2

                vloss = F.binary_cross_entropy_with_logits(logits, prefs)
                val_losses.append(vloss.item())
                val_accs.append(((logits>0).float()==prefs).float().mean().item())
        avg_val_acc = float(np.mean(val_accs))
        avg_val_loss = float(np.mean(val_losses))
        print(f"val_loss={avg_val_loss:.4f} | val_acc={avg_val_acc:.4f}")

        if logger is not None:
            logger.record("reward_model/train_loss", avg_train_loss, exclude=("stdout",))
            logger.record("reward_model/train_acc", avg_train_acc, exclude=("stdout",))
            logger.record("reward_model/val_loss", avg_val_loss, exclude=("stdout",))
            logger.record("reward_model/val_acc", avg_val_acc, exclude=("stdout",))
            logger.dump(iteration)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    rm.eval()
    rm.to('cpu')
    return rm, iteration
        

        