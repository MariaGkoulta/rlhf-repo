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
    model,
    train_dataset,
    val_dataset,
    device,
    epochs,
    patience,
    optimizer,
    regularization_weight,
    logger,
    iteration
):
    """
    Trains the reward model using separate training and validation datasets.
    """
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for s1, a1, s2, a2, pref in train_loader:
            s1, a1, s2, a2, pref = s1.to(device), a1.to(device), s2.to(device), a2.to(device), pref.to(device)
            
            optimizer.zero_grad()
            r1 = model(s1, a1).sum(dim=1)
            r2 = model(s2, a2).sum(dim=1)
            
            logits = r1 - r2
            loss = nn.BCEWithLogitsLoss()(logits, pref)
            
            # L2 regularization on rewards
            l2_reg = regularization_weight * (torch.norm(r1) + torch.norm(r2))
            total_loss = loss + l2_reg
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

            train_preds = (logits > 0).float()
            train_correct += (train_preds == pref).sum().item()
            train_total += pref.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total


        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for s1, a1, s2, a2, pref in val_loader:
                s1, a1, s2, a2, pref = s1.to(device), a1.to(device), s2.to(device), a2.to(device), pref.to(device)
                r1 = model(s1, a1).sum(dim=1)
                r2 = model(s2, a2).sum(dim=1)
                logits = r1 - r2
                val_loss += nn.BCEWithLogitsLoss()(logits, pref).item()
                
                preds = (logits > 0).float()
                correct += (preds == pref).sum().item()
                total += pref.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if logger:
            logger.record("reward_model/train_loss", avg_train_loss, exclude='stdout')
            logger.record("reward_model/train_accuracy", train_acc, exclude='stdout')
            logger.record("reward_model/val_loss", avg_val_loss, exclude='stdout')
            logger.record("reward_model/val_accuracy", val_acc, exclude='stdout')
            logger.dump(step=iteration)
            iteration += 1

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), 'best_reward_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, iteration
        

        