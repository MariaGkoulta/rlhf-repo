import torch
import random
from reward import RewardModel, train_reward_model_batched

class RewardModelEnsemble:
    """
    An ensemble of reward models for uncertainty estimation and active preference selection.
    """
    def __init__(self, obs_dim, act_dim, n_models=5, device='cpu'):
        self.n_models = n_models
        self.device = device
        self.models = [
            RewardModel(obs_dim, act_dim).to(device)
            for _ in range(n_models)
        ]

    def train_ensemble(self, pref_dataset, **train_kwargs):
        """
        Trains each model in the ensemble on a bootstrap sample of the preference dataset.
        """
        for i, model in enumerate(self.models):
            indices = torch.randint(0, len(pref_dataset), (len(pref_dataset),))
            bootstrap_dataset = torch.utils.data.Subset(pref_dataset, indices)
            print(f"Training model {i+1}/{self.n_models} on bootstrap sample of size {len(bootstrap_dataset)}")
            train_reward_model_batched(
                model, bootstrap_dataset, **train_kwargs
            )

    def predict_rewards(self, obs, acts):
        """
        Returns a list of reward predictions from each model in the ensemble.
        """
        with torch.no_grad():
            preds = []
            for model in self.models:
                pred = model.predict_reward(obs, acts)
                preds.append(pred)
            return preds

    def select_pairs_by_disagreement(self, candidate_pairs, top_k=50):
        """
        Selects preference pairs with the highest ensemble disagreement (variance).
        Args:
            candidate_pairs: List of (clip1, clip2) tuples.
            top_k: Number of pairs to select.
        Returns:
            List of selected (clip1, clip2) tuples.
        """
        disagreements = []
        for c1, c2 in candidate_pairs:
            # Compute sum of rewards for each model in the ensemble
            c1_rewards = [float(torch.sum(model.predict_reward(
                torch.tensor(c1["obs"]).float().to(self.device),
                torch.tensor(c1["acts"]).float().to(self.device)
            ))) for model in self.models]
            c2_rewards = [float(torch.sum(model.predict_reward(
                torch.tensor(c2["obs"]).float().to(self.device),
                torch.tensor(c2["acts"]).float().to(self.device)
            ))) for model in self.models]
            prefs = [int(r1 > r2) for r1, r2 in zip(c1_rewards, c2_rewards)]
            disagreement = torch.var(torch.tensor(prefs, dtype=torch.float)).item()
            disagreements.append(disagreement)
        top_indices = sorted(range(len(disagreements)), key=lambda i: -disagreements[i])[:top_k]
        selected_pairs = [candidate_pairs[i] for i in top_indices]
        return selected_pairs