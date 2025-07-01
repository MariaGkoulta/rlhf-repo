# filepath: /rlhf_reacher/src/reward_ensemble.py
import numpy as np
import torch
from torch.utils.data import Subset
from reward import RewardModel, train_reward_model_batched
import random

from preferences import annotate_pairs

class RewardEnsemble:
    def __init__(self, obs_dim, act_dim, num_models=5, dropout_prob=0.1):
        self.models = [RewardModel(obs_dim, act_dim, dropout_prob) for _ in range(num_models)]

    def forward(self, states, actions):
        predictions = torch.stack([model(states, actions) for model in self.models])
        mean_predictions = predictions.mean(dim=0)
        variance_predictions = predictions.var(dim=0)
        return mean_predictions, variance_predictions

    def predict_reward(self, obs, action):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        mean_reward, variance_reward = self.forward(s, a)
        return mean_reward.item(), variance_reward.item()
    
    def train_ensemble(self, pref_dataset, batch_size=64, epochs=20, val_frac=0.1, patience=10, optimizer=None, optimizer_lr=1e-3, optimizer_wd=1e-4, device='cpu', regularization_weight=1e-4, logger=None, iteration=0):
        dataset_size = len(pref_dataset)
        for i, model in enumerate(self.models):
            indices = np.random.choice(dataset_size, dataset_size, replace=True)
            bootstrap_dataset = Subset(pref_dataset, indices)
            print(f"Training model {i+1}/{len(self.models)} on bootstrap sample")
            train_reward_model_batched(
                model, bootstrap_dataset, batch_size=batch_size, epochs=epochs,
                val_frac=val_frac, patience=patience, optimizer=optimizer,
                optimizer_lr=optimizer_lr, optimizer_wd=optimizer_wd,
                device=device, regularization_weight=regularization_weight,
                logger=logger, iteration=iteration
            )

    def state_dict(self):
        return [model.state_dict() for model in self.models]

    def load_state_dict(self, state_dicts):
        for model, sd in zip(self.models, state_dicts):
            model.load_state_dict(sd)

def select_high_variance_pairs(clips, reward_ensemble, num_pairs, min_gap):
    candidate_pairs = []
    num_candidates = min(len(clips) * (len(clips) - 1) // 2, num_pairs * 10)

    indices = list(range(len(clips)))
    while len(candidate_pairs) < num_candidates:
        i, j = random.sample(indices, 2)
        c1, c2 = clips[i], clips[j]
        if abs(sum(c1["rews"]) - sum(c2["rews"])) >= min_gap:
            candidate_pairs.append((c1, c2))
        if len(candidate_pairs) >= num_candidates:
            break
    s1_batch = torch.tensor(np.array([p[0]["obs"] for p in candidate_pairs]), dtype=torch.float32)
    a1_batch = torch.tensor(np.array([p[0]["acts"] for p in candidate_pairs]), dtype=torch.float32)
    s2_batch = torch.tensor(np.array([p[1]["obs"] for p in candidate_pairs]), dtype=torch.float32)
    a2_batch = torch.tensor(np.array([p[1]["acts"] for p in candidate_pairs]), dtype=torch.float32)

    variances = []
    with torch.no_grad():
        preds1_all_models = torch.stack([model(s1_batch, a1_batch).sum(dim=-1) for model in reward_ensemble.models])
        preds2_all_models = torch.stack([model(s2_batch, a2_batch).sum(dim=-1) for model in reward_ensemble.models])
        
        preferences = (preds1_all_models > preds2_all_models).float() # (num_models, num_candidates)
        variances = torch.var(preferences, dim=0).cpu().numpy()

    top_indices = np.argsort(-variances)[:num_pairs]
    selected_pairs = [candidate_pairs[i] for i in top_indices]
    
    prefs, _, _ = annotate_pairs(selected_pairs, min_gap=min_gap)
    return prefs