# filepath: /rlhf_reacher/src/reward_ensemble.py
import numpy as np
import torch
from torch.utils.data import Subset
from reward import RewardModel, train_reward_model_batched
import random
from itertools import combinations

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
    
    def train_ensemble(
            self,
            train_dataset,
            val_dataset,
            device,
            epochs,
            patience,
            optimizer_lr,
            optimizer_wd,
            regularization_weight,
            logger,
            iteration,
            feedback_type=None
        ):
        dataset_size = len(train_dataset)
        for i, model in enumerate(self.models):
            indices = np.random.choice(dataset_size, dataset_size, replace=True)
            bootstrap_dataset = Subset(train_dataset, indices)
            print(f"Training model {i+1}/{len(self.models)} on bootstrap sample")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr, weight_decay=optimizer_wd)
            
            train_reward_model_batched(
                model,
                train_dataset=bootstrap_dataset,
                val_dataset=val_dataset,
                device=device,
                epochs=epochs,
                patience=patience,
                optimizer=optimizer,
                regularization_weight=regularization_weight,
                logger=logger,
                iteration=iteration,
                feedback_type=feedback_type

            )

    def state_dict(self):
        return [model.state_dict() for model in self.models]

    def load_state_dict(self, state_dicts):
        for model, sd in zip(self.models, state_dicts):
            model.load_state_dict(sd)

def select_high_variance_pairs(
    clips,
    reward_ensemble,
    num_pairs, min_gap,
    uncertainty_method='bald',
    logger=None,
    iteration=None):
    
    candidate_pairs = []
    num_candidates = min(len(clips) * (len(clips) - 1) // 2, num_pairs * 20)
    if len(clips) < 2:
        return []
    indices = list(range(len(clips)))    
    all_possible_pairs = list(combinations(indices, 2))
    random.shuffle(all_possible_pairs)
    for i, j in all_possible_pairs:
        if len(candidate_pairs) >= num_candidates:
            break
        c1, c2 = clips[i], clips[j]
        if "rews" in c1 and "rews" in c2 and abs(sum(c1["rews"]) - sum(c2["rews"])) >= min_gap:
            candidate_pairs.append((c1, c2))
        elif "rews" not in c1 or "rews" not in c2:
             candidate_pairs.append((c1, c2))
    s1_batch = torch.tensor(np.array([p[0]["obs"] for p in candidate_pairs]), dtype=torch.float32)
    a1_batch = torch.tensor(np.array([p[0]["acts"] for p in candidate_pairs]), dtype=torch.float32)
    s2_batch = torch.tensor(np.array([p[1]["obs"] for p in candidate_pairs]), dtype=torch.float32)
    a2_batch = torch.tensor(np.array([p[1]["acts"] for p in candidate_pairs]), dtype=torch.float32)
    variances = []
    with torch.no_grad():
        preds1_all_models = torch.stack([model(s1_batch, a1_batch).sum(dim=-1) for model in reward_ensemble.models])
        preds2_all_models = torch.stack([model(s2_batch, a2_batch).sum(dim=-1) for model in reward_ensemble.models])
        if uncertainty_method == 'bald':
            return_diffs = preds1_all_models - preds2_all_models
            variances = torch.var(return_diffs, dim=0).cpu().numpy()
        elif uncertainty_method == 'softmax':
            probs1 = torch.softmax(torch.stack([preds1_all_models, preds2_all_models], dim=-1), dim=-1)
            variances = torch.var(probs1[:, :, 0], dim=0).cpu().numpy()
        else:
            preferences = (preds1_all_models > preds2_all_models).float() # (num_models, num_candidates)
            variances = torch.var(preferences, dim=0).cpu().numpy()
    
    if logger and iteration is not None and len(variances) > 0:
        logger.record("active_learning/ensemble_variance_mean", np.mean(variances))
        logger.record("active_learning/ensemble_variance_var", np.var(variances))

    top_indices = np.argsort(-variances)[:num_pairs]
    selected_pairs = [candidate_pairs[i] for i in top_indices]
    prefs, _, _ = annotate_pairs(selected_pairs, min_gap=min_gap)
    return prefs