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
    rating_scale=None,
    gamma=0.99
):
    """
    Trains the reward model using separate training and validation datasets.
    Supports both preference and evaluative feedback types within the same batch.
    """
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=mixed_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=mixed_collate_fn)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_pref_correct = 0
        train_pref_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Unpack batch
            s1, a1, s2, a2, prefs, eval_states, eval_actions, eval_ratings = batch
            
            total_loss = 0
            
            # Preference loss
            if prefs.size(0) > 0:

                s1, a1, s2, a2, prefs = s1.to(device), a1.to(device), s2.to(device), a2.to(device), prefs.to(device)
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
                loss = nn.BCEWithLogitsLoss()(logits, prefs)
                # L2 regularization on rewards
                l2_reg = regularization_weight * (torch.norm(r1) + torch.norm(r2))
                total_loss = loss + l2_reg
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
                train_preds = (logits > 0).float()
                train_pref_correct += (train_preds == prefs).sum().item()
                train_pref_total += prefs.size(0)

            # Evaluative loss
            if eval_ratings.size(0) > 0:


                eval_states, eval_actions, eval_ratings = eval_states.to(device), eval_actions.to(device), eval_ratings.to(device)
                optimizer.zero_grad()
                
                if model.probabilistic:
                    # Use Gaussian NLL loss for evaluative feedback
                    mean_rewards, var_rewards = model(eval_states, eval_actions)
                    predicted_mean = mean_rewards.sum(dim=1)
                    predicted_var = var_rewards.sum(dim=1)
                    # Gaussian Negative Log Likelihood Loss
                    loss = nn.GaussianNLLLoss()(predicted_mean, eval_ratings, predicted_var)
                else:
                    predicted_segment_rewards = model(eval_states, eval_actions).sum(dim=1)
                    loss = nn.MSELoss()(predicted_segment_rewards, eval_ratings)

                total_loss = loss
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
                train_total += eval_ratings.size(0)

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = train_pref_correct / train_pref_total if train_pref_total > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0
        val_pref_correct = 0
        val_pref_total = 0
        with torch.no_grad():
            for batch in val_loader:
                s1, a1, s2, a2, prefs, eval_states, eval_actions, eval_ratings = batch
                
                total_val_loss = 0

                if prefs.size(0) > 0:
                    s1, a1, s2, a2, prefs = s1.to(device), a1.to(device), s2.to(device), a2.to(device), prefs.to(device)
                    if model.probabilistic:
                        r1_mean, _ = model(s1, a1)
                        r2_mean, _ = model(s2, a2)
                        r1 = r1_mean.sum(dim=1)
                        r2 = r2_mean.sum(dim=1)
                    else:
                        r1 = model(s1, a1).sum(dim=1)
                        r2 = model(s2, a2).sum(dim=1)
                    logits = r1 - r2
                    total_val_loss += nn.BCEWithLogitsLoss()(logits, prefs).item()
                    
                    preds = (logits > 0).float()
                    val_pref_correct += (preds == prefs).sum().item()
                    val_pref_total += prefs.size(0)

                if eval_ratings.size(0) > 0:
                    eval_states, eval_actions, eval_ratings = eval_states.to(device), eval_actions.to(device), eval_ratings.to(device)
                    per_step_rewards = model(eval_states, eval_actions)
                    predicted_segment_rewards = per_step_rewards.sum(dim=1)
                    total_val_loss += nn.MSELoss()(predicted_segment_rewards, eval_ratings).item()
                
                val_loss += total_val_loss
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = val_pref_correct / val_pref_total if val_pref_total > 0 else 0.0

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

def mixed_collate_fn(batch):
    """
    Custom collate function to handle batches of mixed preference and evaluative data.
    """
    # Separate items by type
    pref_items = [item for item in batch if len(item) == 5]
    eval_items = [item for item in batch if len(item) == 2]

    # Process preference data
    if pref_items:
        s1, a1, s2, a2, prefs = zip(*pref_items)
        s1 = torch.stack(s1)
        a1 = torch.stack(a1)
        s2 = torch.stack(s2)
        a2 = torch.stack(a2)
        prefs = torch.tensor(prefs, dtype=torch.float32)
    else:
        s1, a1, s2, a2, prefs = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    # Process evaluative data
    if eval_items:
        clips, ratings = zip(*eval_items)
        eval_states = torch.stack([item['obs'] for item in clips])
        eval_actions = torch.stack([item['acts'] for item in clips])
        eval_ratings = torch.tensor(ratings, dtype=torch.float32)
    else:
        eval_states, eval_actions, eval_ratings = torch.empty(0), torch.empty(0), torch.empty(0)

    return s1, a1, s2, a2, prefs, eval_states, eval_actions, eval_ratings

