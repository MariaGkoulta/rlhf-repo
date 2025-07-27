import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

class RewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, dropout_prob=0.1, hidden=None, probabilistic=False):
        super().__init__()
        self.probabilistic = probabilistic
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 64), nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
        )
        
        if self.probabilistic:
            # Separate heads for mean and variance
            self.mean_head = nn.Linear(64, 1)
            self.var_head = nn.Sequential(
                nn.Linear(64, 1),
                nn.Softplus()  # Ensures positive variance
            )
        else:
            # Original deterministic head
            self.net = nn.Sequential(
                self.feature_net,
                nn.Linear(64, 1)
            )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        
        if self.probabilistic:
            features = self.feature_net(x)
            mean = self.mean_head(features).squeeze(-1)
            var = self.var_head(features).squeeze(-1)
            return mean, var
        else:
            return self.net(x).squeeze(-1)

    def predict_reward(self, obs, action):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.probabilistic:
                mean, var = self.forward(s, a)
                return mean.item()  # Return mean for compatibility
            else:
                return self.forward(s, a).item()
        
def calculate_discounted_reward_for_predictions(rewards, gamma):
    """
    Calculates the discounted reward for a batch of trajectories.
    """
    seq_len = rewards.shape[1]
    discounts = torch.pow(gamma, torch.arange(seq_len, device=rewards.device)).unsqueeze(0)
    return (rewards * discounts).sum(dim=1)
        
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
    iteration,
    feedback_type="preference",
    rating_scale=None,
    gamma=0.99
):
    """
    Trains the reward model using separate training and validation datasets.
    Supports both preference and evaluative feedback types.
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
        
        for batch in train_loader:
            if feedback_type == "preference":
                s1, a1, s2, a2, pref = batch
                s1, a1, s2, a2, pref = s1.to(device), a1.to(device), s2.to(device), a2.to(device), pref.to(device)
                optimizer.zero_grad()
                
                if model.probabilistic:
                    # Use only mean for preferences, ignore variance
                    r1_mean, r1_var = model(s1, a1)
                    r2_mean, r2_var = model(s2, a2)
                    r1 = r1_mean.sum(dim=1)
                    r2 = r2_mean.sum(dim=1)
                else:
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
                
            elif feedback_type == "evaluative":
                states, actions, ratings = batch
                states, actions, ratings = states.to(device), actions.to(device), ratings.to(device)
                optimizer.zero_grad()
                
                if model.probabilistic:
                    # Use Gaussian NLL loss for evaluative feedback
                    mean_rewards, var_rewards = model(states, actions)
                    predicted_mean = mean_rewards.sum(dim=1)
                    predicted_var = var_rewards.sum(dim=1)
                    # Gaussian Negative Log Likelihood Loss
                    loss = nn.GaussianNLLLoss()(predicted_mean, ratings, predicted_var)
                else:
                    predicted_segment_rewards = model(states, actions).sum(dim=1)
                    loss = nn.MSELoss()(predicted_segment_rewards, ratings)
                    
                total_loss = loss
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
                train_total += ratings.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        if feedback_type == "preference":
            train_acc = train_correct / train_total
        else:
            train_acc = 0.0

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if feedback_type == "preference":
                    s1, a1, s2, a2, pref = batch
                    s1, a1, s2, a2, pref = s1.to(device), a1.to(device), s2.to(device), a2.to(device), pref.to(device)
                    
                    if model.probabilistic:
                        r1_mean, r1_var = model(s1, a1)
                        r2_mean, r2_var = model(s2, a2)
                        r1 = r1_mean.sum(dim=1)
                        r2 = r2_mean.sum(dim=1)
                    else:
                        r1 = model(s1, a1).sum(dim=1)
                        r2 = model(s2, a2).sum(dim=1)
                        
                    logits = r1 - r2
                    val_loss += nn.BCEWithLogitsLoss()(logits, pref).item()
                    
                    preds = (logits > 0).float()
                    correct += (preds == pref).sum().item()
                    total += pref.size(0)
                    
                elif feedback_type == "evaluative":
                    states, actions, ratings = batch
                    states, actions, ratings = states.to(device), actions.to(device), ratings.to(device)
                    
                    if model.probabilistic:
                        mean_rewards, var_rewards = model(states, actions)
                        predicted_mean = mean_rewards.sum(dim=1)
                        predicted_var = var_rewards.sum(dim=1)
                        val_loss += nn.GaussianNLLLoss()(predicted_mean, ratings, predicted_var).item()
                    else:
                        per_step_rewards = model(states, actions)
                        predicted_segment_rewards = per_step_rewards.sum(dim=1)
                        val_loss += nn.MSELoss()(predicted_segment_rewards, ratings).item()
                    total += ratings.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        if feedback_type == "preference":
            val_acc = correct / total
        else:
            val_acc = 0.0

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
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, iteration

