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

from gymnasium.wrappers import RecordVideo
import os

TOTAL_ITERS        = 8
CLIPS_PER_UPDATE   = 50000
SEGMENT_LEN        = 40
TIMESTEPS_PER_ITER = 250000

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
        # Convert to tensor once upon addition
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
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)

    def predict_reward(self, obs, action):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.forward(s, a).item()

# def train_reward_model(rm, dataset, epochs=20, lr=1e-3):
#     opt = torch.optim.Adam(rm.parameters(), lr=lr)
#     for _ in range(epochs):
#         for clip1, clip2, pref in dataset:
#             s1, a1 = torch.from_numpy(np.stack(clip1["obs"])).float(), torch.from_numpy(np.stack(clip1["acts"])).float()
#             s2, a2 = torch.from_numpy(np.stack(clip2["obs"])).float(), torch.from_numpy(np.stack(clip2["acts"])).float()
#             r1, r2 = rm(s1,a1).sum(), rm(s2,a2).sum()
#             logits    = (r1 - r2).unsqueeze(0)                     # shape [1]
#             targets   = torch.tensor([pref], dtype=torch.float32)  # 0.0 or 1.0
#             loss      = F.binary_cross_entropy_with_logits(logits, targets)
#             opt.zero_grad();
#             loss.backward();
#             opt.step()
#     return rm


def train_reward_model_with_val(
    rm,
    dataset,
    epochs=20,
    lr=1e-3,
    val_frac=0.1,
    patience=5,
):
    train_data, val_data = train_test_split(dataset, test_size=val_frac, random_state=42)
    opt = torch.optim.Adam(
        rm.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        rm.train()
        train_losses = []
        random.shuffle(train_data)
        for clip1, clip2, pref in train_data:
            s1 = torch.from_numpy(np.stack(clip1["obs"])).float()
            a1 = torch.from_numpy(np.stack(clip1["acts"])).float()
            s2 = torch.from_numpy(np.stack(clip2["obs"])).float()
            a2 = torch.from_numpy(np.stack(clip2["acts"])).float()

            r1 = rm(s1, a1).sum()
            r2 = rm(s2, a2).sum()
            logits  = (r1 - r2).unsqueeze(0)  
            target  = torch.tensor([pref], dtype=torch.float32)

            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())
        avg_train = float(np.mean(train_losses))

        rm.eval()
        val_losses = []
        with torch.no_grad():
            for clip1, clip2, pref in val_data:
                s1 = torch.from_numpy(np.stack(clip1["obs"])).float()
                a1 = torch.from_numpy(np.stack(clip1["acts"])).float()
                s2 = torch.from_numpy(np.stack(clip2["obs"])).float()
                a2 = torch.from_numpy(np.stack(clip2["acts"])).float()

                r1 = rm(s1, a1).sum()
                r2 = rm(s2, a2).sum()
                logits = (r1 - r2).unsqueeze(0)
                target = torch.tensor([pref], dtype=torch.float32)

                loss = F.binary_cross_entropy_with_logits(logits, target)
                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d} — train_loss: {avg_train:.4f}, val_loss: {avg_val:.4f}")

        # — EARLY STOPPING —
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Stopping early: no improvement in {patience} epochs.")
                break

    return rm

def collect_clips(policy, num_clips, segment_len):
    clips = []
    raw   = gym.make("Reacher-v4", render_mode="rgb_array")
    for _ in range(num_clips):
        obs, _ = raw.reset()
        traj = {"obs": [], "acts": []}
        done = False
        while not done:
            act, _ = policy.predict(obs, deterministic=False)
            traj["obs"].append(obs)
            traj["acts"].append(act.reshape(-1))
            obs, _, terminated, truncated, _ = raw.step(act)
            done = terminated or truncated
        if len(traj["acts"]) <= segment_len:
            seg = traj
        else:
            start = np.random.randint(0, len(traj["acts"]) - segment_len)
            seg = {
              "obs":  traj["obs"][start:start+segment_len],
              "acts": traj["acts"][start:start+segment_len]
            }
        clips.append(seg)
    raw.close()
    return clips

def eval_clip(c, env):
    o, _ = env.reset()
    tot  = 0
    for a in c["acts"]:
        a = np.asarray(a).reshape(-1)
        o, r, term, trunc, _ = env.step(a)
        tot += r
        if term or trunc:
            break
    return tot

# ── 4) annotate by replaying each clip in a raw env ──
def annotate_preferences(clips, num_clips=10000, min_gap=1.0):
    prefs = []
    gaps = []
    true_env = gym.make("Reacher-v4")
    for _ in range(num_clips):
        c1, c2 = random.sample(clips, 2)
        r1 = eval_clip(c1, true_env)
        r2 = eval_clip(c2, true_env)
        gaps.append(abs(r1 - r2))
        if abs(r1 - r2) < min_gap:
            continue
        p  = 1 if r1 > r2 else 0
        prefs.append((c1, c2, p))
    true_env.close()
    plt.hist(gaps, bins=50); plt.title("Return Gaps")
    return prefs

# ── 5) bootstrap PPO on true reward (no wrapper needed) ──────────────────────
raw_env = gym.make("Reacher-v4", render_mode="rgb_array")
policy  = PPO("MlpPolicy", raw_env, 
              verbose=1, n_steps=2048, batch_size=64, tensorboard_log="./logs/ppo_reacher/")
policy.learn(total_timesteps=TIMESTEPS_PER_ITER)
raw_env.close()

# ── 6) initial preference‐based reward model ────────────────────────────────
init_clips      = collect_clips(policy, CLIPS_PER_UPDATE, SEGMENT_LEN)
preference_data = annotate_preferences(init_clips)
obs_dim, act_dim= policy.observation_space.shape[0], policy.action_space.shape[0]
reward_model    = RewardModel(obs_dim, act_dim)

s = np.random.randn(obs_dim).astype(np.float32)
s = torch.tensor(s).unsqueeze(0)
a = np.random.randn(act_dim).astype(np.float32)
a = torch.tensor(a).unsqueeze(0)

reward_before = reward_model.predict_reward(s, a)
print(f"Initial reward model prediction: {reward_before}")

reward_model = train_reward_model_with_val(
        reward_model,
        preference_data,
        epochs=50,
        lr=3e-3,
        val_frac=0.1,
        patience=7
)

reward_after = reward_model(s, a)
print(f"Trained reward model prediction: {reward_after}")

# ── 7) now your RLHF loop with wrapped & normalized vec‐env ─────────────────
# def make_wrapped():
#     raw = gym.make("Reacher-v4", render_mode="rgb_array")
#     return LearnedRewardEnv(raw, reward_model)

# vec_env = VecNormalize(
#     DummyVecEnv([make_wrapped]),
#     norm_obs=True, norm_reward=True
# )
# # vec_env = DummyVecEnv([make_wrapped])
# policy.set_env(vec_env)

def make_wrapped():
    env = gym.make("Reacher-v4", render_mode="rgb_array")
    env = Monitor(env)
    return LearnedRewardEnv(env, reward_model)

vec_env = DummyVecEnv([make_wrapped])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
vec_env = VecMonitor(vec_env)
policy.set_env(vec_env)

for iteration in range(1, TOTAL_ITERS):
    policy.learn(total_timesteps=TIMESTEPS_PER_ITER, reset_num_timesteps=False)

    # collect & annotate new clips
    clips    = collect_clips(policy, CLIPS_PER_UPDATE, SEGMENT_LEN)

    # plot clips reward distribution
    rewards  = [eval_clip(c, vec_env.envs[0]) for c in clips]
    plt.hist(rewards, bins=20)
    plt.title(f"Iteration {iteration} - Clip Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.savefig(f"clip_rewards_iter_{iteration}.png")
    plt.close()

    new_prefs= annotate_preferences(clips)
    preference_data.extend(new_prefs)

    reward_model = train_reward_model_with_val(
        reward_model,
        preference_data,
        epochs=50,
        lr=3e-3,
        val_frac=0.1,
        patience=7
    )
    for sub in vec_env.envs:
        if isinstance(sub, LearnedRewardEnv):
            sub.reward_model = reward_model
    policy.set_env(vec_env)
    reward_after_rlhf = reward_model(s, a)
    print(f"Trained reward model prediction after rlhf: {reward_after_rlhf}")

    ep_true_list    = vec_env.get_attr("ep_true")[0]
    ep_learned_list = vec_env.get_attr("ep_learned")[0]

    true_returns    = np.array(ep_true_list)
    learned_returns = np.array(ep_learned_list)
    corr = np.corrcoef(true_returns, learned_returns)[0,1]
    print(f"Pearson r (true vs learned): {corr:.3f}")
    r, pval = pearsonr(true_returns, learned_returns)
    print(f"r = {r:.3f}, p-value = {pval:.3e}")


# ── 8) save the final model and evaluate it ─────────────────────────────────
print("Evaluating final policy and recording video...")
video_folder = "videos/final_policy_evaluation"
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