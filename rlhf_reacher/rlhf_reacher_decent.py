import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader, random_split
from stable_baselines3.common.callbacks import BaseCallback

from tqdm import tqdm

from gymnasium.wrappers import RecordVideo
import os
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}")

TOTAL_ITERS        = 10
CLIPS_PER_UPDATE   = 100_000
NUM_PAIRS          = 50_000
SEGMENT_LEN        = 50
INITIAL_CLIPS_SIZE = 100_000
INITIAL_PREF_DS_SIZE = 50_000
TIMESTEPS_PER_ITER = 100000
INITIAL_POLICY_TIMESTEPS = 50_000

class TrueRewardCallback(BaseCallback):
    """
    After each rollout, fetch the ep_true lists from your LearnedRewardEnv,
    compute their mean, record it under rollout/ep_true_mean, and clear them.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_true_returns = 0
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        true_returns_collected = []
        for ep_list in self.training_env.get_attr("ep_true"):
            true_returns_collected.extend(ep_list)
        learned_returns_collected = []
        for ep_list in self.training_env.get_attr("ep_learned"):
            learned_returns_collected.extend(ep_list)
        if len(true_returns_collected) > 0:
            mean_true = float(np.mean(true_returns_collected))
            self.logger.record("rollout/ep_true_mean", mean_true)
        if len(learned_returns_collected) > 0:
            mean_learned = float(np.mean(learned_returns_collected))
            self.logger.record("rollout/ep_learned_mean", mean_learned)
        if len(true_returns_collected) > 1 and len(learned_returns_collected) > 1 and len(true_returns_collected) == len(learned_returns_collected):
            true_returns_np = np.array(true_returns_collected)
            learned_returns_np = np.array(learned_returns_collected)
            if np.std(true_returns_np) > 0 and np.std(learned_returns_np) > 0:
                correlation, p_value = pearsonr(true_returns_np, learned_returns_np)
                self.logger.record("rollout/pearson_r_true_vs_learned", correlation)
                self.logger.record("rollout/pearson_p_value", p_value)
            else:
                self.logger.record("rollout/pearson_r_true_vs_learned", np.nan) # Or 0, or skip logging
                self.logger.record("rollout/pearson_p_value", np.nan)
        elif len(true_returns_collected) > 0 or len(learned_returns_collected) > 0 : # Log NaN if data is insufficient/mismatched
             self.logger.record("rollout/pearson_r_true_vs_learned", np.nan)
             self.logger.record("rollout/pearson_p_value", np.nan)
        for sub in self.training_env.envs:
            if hasattr(sub, "ep_true"):
                sub.ep_true.clear()
                sub.ep_learned.clear()

    def _on_step(self) -> bool:
        return True

class PreferenceDataset(Dataset):
    """
    Stores preference-labeled clip pairs as tensors, converting each clip only once on addition.
    """
    def __init__(self, device='cpu'):
        self.s1 = []
        self.a1 = []
        self.s2 = []
        self.a2 = []
        self.prefs = []
        self.device = device

    def add(self, clip1, clip2, pref):
        s1 = torch.tensor(np.stack(clip1['obs']), dtype=torch.float32, device=self.device)
        a1 = torch.tensor(np.stack(clip1['acts']), dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack(clip2['obs']), dtype=torch.float32, device=self.device)
        a2 = torch.tensor(np.stack(clip2['acts']), dtype=torch.float32, device=self.device)
        p  = torch.tensor(pref, dtype=torch.float32, device=self.device)
        self.s1.append(s1)
        self.a1.append(a1)
        self.s2.append(s2)
        self.a2.append(a2)
        self.prefs.append(p)

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx):
        return self.s1[idx], self.a1[idx], self.s2[idx], self.a2[idx], self.prefs[idx]

class LearnedRewardEnv(gym.Wrapper):
    """Gymnasium new API:
       reset() → (obs, info)
       step()  → (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model
        self.prev_obs = None
        self.ep_true = []
        self.ep_learned = []

    def reset(self, **kwargs):
        obs, info     = super().reset(**kwargs)
        self.prev_obs = obs
        self._true_sum = 0.0
        self._learned_sum = 0.0
        return obs, info


    def step(self, action):
        obs, true_r, terminated, truncated, info = super().step(action)
        learned_r = self.reward_model.predict_reward(self.prev_obs, action)
        self._true_sum += true_r
        self._learned_sum += learned_r
        done = terminated or truncated
        if done:
            info = info.copy()
            info["episode"] = {
                "r_true": self._true_sum,
                "r_learned": self._learned_sum,
            }
            self.ep_true.append(self._true_sum)
            self.ep_learned.append(self._learned_sum)
        self.prev_obs = obs
        return obs, learned_r, terminated, truncated, info

class RewardModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
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
    rm,
    pref_dataset: PreferenceDataset,
    batch_size=64,
    epochs=20,
    lr=1e-3,
    val_frac=0.1,
    patience=5,
    device='cpu'
):
    """
    Trains RewardModel using mini-batches from a tensor-based PreferenceDataset.
    Assumes each clip has shape [T, obs_dim], [T, act_dim].
    """
    rm.to(device)

    total = len(pref_dataset)
    val_size = int(total * val_frac)
    train_size = total - val_size
    train_ds, val_ds = random_split(pref_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(rm.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        # --- Training ---
        rm.train()
        train_losses = []
        train_accs = []
        for s1, a1, s2, a2, prefs in train_loader:
            N, T, obs_dim = s1.shape
            _, _, act_dim = a1.shape

            s1_flat = s1.view(N * T, obs_dim)
            a1_flat = a1.view(N * T, act_dim)
            s2_flat = s2.view(N * T, obs_dim)
            a2_flat = a2.view(N * T, act_dim)
            r1 = rm(s1_flat, a1_flat).view(N, T).sum(dim=1)
            r2 = rm(s2_flat, a2_flat).view(N, T).sum(dim=1)

            logits = r1 - r2
            bce    = F.binary_cross_entropy_with_logits(logits, prefs)
            reg    = 1e-3 * (r1.pow(2).mean() + r2.pow(2).mean())
            loss   = bce + reg
            acc = (logits > 0).float().eq(prefs).float().mean()
            train_accs.append(acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        avg_train_acc = float(np.mean(train_accs))
        
        rm.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for s1, a1, s2, a2, prefs in val_loader:
                N, T, obs_dim = s1.shape
                _, _, act_dim = a1.shape

                s1_flat = s1.view(N * T, obs_dim)
                a1_flat = a1.view(N * T, act_dim)
                s2_flat = s2.view(N * T, obs_dim)
                a2_flat = a2.view(N * T, act_dim)

                r1 = rm(s1_flat, a1_flat).view(N, T).sum(dim=1)
                r2 = rm(s2_flat, a2_flat).view(N, T).sum(dim=1)

                logits = r1 - r2
                loss   = F.binary_cross_entropy_with_logits(logits, prefs)
                val_losses.append(loss.item())
                val_acc = (logits > 0).float().eq(prefs).float().mean()
                val_accs.append(val_acc.item())

        avg_val_loss = float(np.mean(val_losses))
        avg_val_acc = float(np.mean(val_accs))
        print(f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | train_acc={avg_train_acc:.4f} | val_loss={avg_val_loss:.4f} | val_acc={avg_val_acc:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stop at epoch {epoch}: no improvement for {patience} epochs.")
                break

    rm.to('cpu')
    return rm


def annotate_preferences_old(
    clips,
    num_pairs=NUM_PAIRS,
    min_gap=0,
    top_frac=0.75,
    half_frac=0.5
):
    returns = np.array([sum(c["rews"]) for c in clips])
    thresh = np.percentile(returns, top_frac * 100)
    top_clips = [c for c, r in zip(clips, returns) if r >= thresh]
    
    prefs = []
    diffs = []
    
    n_top_pairs = int(num_pairs * half_frac)
    n_rand_pairs = num_pairs - n_top_pairs
    
    for _ in range(n_top_pairs):
        if len(top_clips) < 2:
            break
        c1, c2 = random.sample(top_clips, 2)
        r1, r2 = sum(c1["rews"]), sum(c2["rews"])
        if abs(r1 - r2) < min_gap:
            continue
        prefs.append((c1, c2, 1 if r1 > r2 else 0))
        diffs.append(abs(r1 - r2))
    
    for _ in range(n_rand_pairs):
        c1, c2 = random.sample(clips, 2)
        r1, r2 = sum(c1["rews"]), sum(c2["rews"])
        if abs(r1 - r2) < min_gap:
            continue
        prefs.append((c1, c2, 1 if r1 > r2 else 0))
        diffs.append(abs(r1 - r2))
    
    combined = list(zip(prefs, diffs))
    random.shuffle(combined)
    prefs, diffs = zip(*combined)
    
    return list(prefs), list(diffs)

def collect_clips(policy, num_clips, segment_len, n_envs: int = 8):
    make_env = lambda: gym.make("Reacher-v4", render_mode=None)
    vec_env  = DummyVecEnv([make_env] * n_envs)
    obs            = vec_env.reset()
    current_trajs  = [{"obs": [], "acts": [], "rews": []} for _ in range(n_envs)]
    clips          = []
    pbar = tqdm(total=num_clips, desc=f"Collecting clips with {n_envs} envs")
    while len(clips) < num_clips:
        actions, _ = policy.predict(obs, deterministic=False)
        next_obs, rewards, dones, _ = vec_env.step(actions)
        for i in range(n_envs):
            current_trajs[i]["obs"].append(obs[i])
            current_trajs[i]["acts"].append(actions[i].reshape(-1))
            current_trajs[i]["rews"].append(rewards[i])
            if dones[i]:
                traj = current_trajs[i]
                if len(traj["acts"]) <= segment_len:
                    seg = traj
                else:
                    start = np.random.randint(0, len(traj["acts"]) - segment_len + 1)
                    seg   = {k: v[start:start + segment_len] for k, v in traj.items()}
                clips.append(seg)
                pbar.update(1)
                current_trajs[i] = {"obs": [], "acts": [], "rews": []}
        obs = next_obs

    # Print reward distribution contained in clips
    rewards = [sum(c["rews"]) for c in clips]
    plt.hist(rewards, bins=20)
    plt.title("Reward Distribution of Collected Clips")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.close()

    # Reward evolution plot
    plt.plot(rewards)
    plt.title("Reward Evolution of Collected Clips")
    plt.xlabel("Clip Index")
    plt.ylabel("Reward")
    plt.close()

    pbar.close()
    vec_env.close()
    return clips[:num_clips]

def clip_return(c):
    return sum(c["rews"])

def annotate_preferences(clips, num_pairs=NUM_PAIRS, min_gap=2.0):
    prefs = []
    reward_differences = []
    for _ in tqdm(range(num_pairs)):
        c1, c2 = random.sample(clips, 2)
        r1, r2 = clip_return(c1), clip_return(c2)
        difference = abs(r1 - r2)
        if abs(r1 - r2) < min_gap:
            continue
        prefs.append((c1, c2, 1 if r1 > r2 else 0))
        reward_differences.append(difference)
    return prefs, reward_differences

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pref_ds = PreferenceDataset(device=device)

# ── Bootstrap PPO on true reward (no wrapper needed) ──────────────────────
raw_env = gym.make("Reacher-v4", render_mode="rgb_array")
policy  = PPO("MlpPolicy", raw_env, 
              verbose=1, n_steps=2048, batch_size=16, tensorboard_log="./logs/ppo_reacher/")
policy.learn(total_timesteps=INITIAL_POLICY_TIMESTEPS)
raw_env.close()

# ── Initial preference‐based reward model ────────────────────────────────
init_clips      = collect_clips(policy, INITIAL_CLIPS_SIZE, SEGMENT_LEN)
preference_data, _reward_diffs = annotate_preferences(init_clips, num_pairs=INITIAL_PREF_DS_SIZE)

for c1, c2, p in preference_data:
    pref_ds.add(c1, c2, p)

obs_dim, act_dim= policy.observation_space.shape[0], policy.action_space.shape[0]
reward_model    = RewardModel(obs_dim, act_dim)

s = np.random.randn(obs_dim).astype(np.float32)
s = torch.tensor(s).unsqueeze(0)
a = np.random.randn(act_dim).astype(np.float32)
a = torch.tensor(a).unsqueeze(0)


reward_model = train_reward_model_batched(
    reward_model,
    pref_ds,
    batch_size=64,
    epochs=50,
    lr=3e-3,
    val_frac=0.1,
    patience=7,
    device=pref_ds.device
)

def make_wrapped():
    env = gym.make("Reacher-v4", render_mode="rgb_array")
    env = Monitor(env)
    return LearnedRewardEnv(env, reward_model)

vec_env = DummyVecEnv([make_wrapped])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
vec_env = VecMonitor(vec_env)
policy.set_env(vec_env)
callback = TrueRewardCallback()

for iteration in range(1, TOTAL_ITERS):
    policy.learn(total_timesteps=TIMESTEPS_PER_ITER, reset_num_timesteps=False, callback=callback)

    # collect & annotate new clips
    clips    = collect_clips(policy, CLIPS_PER_UPDATE, SEGMENT_LEN)

    clip_rewards = [clip_return(c) for c in clips]
    plt.figure(figsize=(10, 6))
    plt.hist(clip_rewards, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Iteration {iteration} - Distribution of Collected Clip Rewards")
    plt.xlabel("Sum of True Rewards per Clip")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plot_path_clips = os.path.join(results_dir, f"iter_{iteration}_clip_rewards_dist.png")
    plt.savefig(plot_path_clips)
    plt.close()
    print(f"Saved clip rewards distribution plot to {plot_path_clips}")

    print("Annotating preferences for new clips...")
    new_prefs, reward_differences= annotate_preferences(clips)


    plt.figure(figsize=(10, 6))
    plt.hist(reward_differences, bins=50, color='lightcoral', edgecolor='black')
    plt.title(f"Iteration {iteration} - Distribution of Reward Differences in Preference Pairs")
    plt.xlabel("Difference in Sum of True Rewards (Clip1 Reward - Clip2 Reward)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plot_path_prefs_diff = os.path.join(results_dir, f"iter_{iteration}_pref_reward_diff_dist.png")
    plt.savefig(plot_path_prefs_diff)
    plt.close()
    print(f"Saved preference reward differences plot to {plot_path_prefs_diff}")

    for clip1, clip2, p in new_prefs:
        pref_ds.add(clip1, clip2, p)

    reward_model = train_reward_model_batched(
        reward_model,
        pref_ds,
        batch_size=64,
        epochs=50,
        lr=3e-3,
        val_frac=0.1,
        patience=7,
        device=pref_ds.device
    )
    for sub in vec_env.envs:
        if isinstance(sub, LearnedRewardEnv):
            sub.reward_model = reward_model
    policy.set_env(vec_env)

# save reward model
reward_model_path = "reward_model_1905.pth"
torch.save(reward_model.state_dict(), reward_model_path)
print(f"Reward model saved to {reward_model_path}")

policy.save("ppo_reacher_1905")

# ── Save the final model and evaluate it ─────────────────────────────────
print("Evaluating final policy and recording video...")
video_folder = os.path.join(results_dir, f"final_policy_evaluation")
os.makedirs(video_folder, exist_ok=True)
eval_env = gym.make("Reacher-v4", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env,
    video_folder=video_folder,
    episode_trigger=lambda ep_id: ep_id == 0,
    name_prefix="final-reacher-eval"
)
obs, _ = eval_env.reset()
done = False
while not done:
    action, _ = policy.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
eval_env.close()
print(f"Final policy evaluation video saved to {video_folder}/")