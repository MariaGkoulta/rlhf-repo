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
import pickle

SEGMENT_LEN        = 50
INITIAL_POLICY_TIMESTEPS = 1000000
TOTAL_TRAINING_TIMESTEPS_FOR_CLIPS = INITIAL_POLICY_TIMESTEPS  # Use existing constant
COLLECTION_INTERVAL_TIMESTEPS = 10000
NUM_CLIPS_PER_COLLECTION = 100
OUTPUT_FILE_PATH = "collected_clips_with_rewards.pkl"


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
                if len(clips) >= num_clips: # Ensure we don't overshoot due to parallel envs
                    break
                current_trajs[i] = {"obs": [], "acts": [], "rews": []}
        if len(clips) >= num_clips: # Check again after processing all envs in a step
            break
        obs = next_obs

    # Print reward distribution contained in clips
    rewards_collected = [sum(c["rews"]) for c in clips[:num_clips]] # Ensure only num_clips are processed
    if rewards_collected: # Check if any rewards were collected to avoid error with empty list
        plt.hist(rewards_collected, bins=20)
        plt.title("Reward Distribution of Collected Clips")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.savefig("collected_clips_reward_distribution.png") # Save the plot
        plt.close()

        # Reward evolution plot
        plt.plot(rewards_collected)
        plt.title("Reward Evolution of Collected Clips")
        plt.xlabel("Clip Index")
        plt.ylabel("Reward")
        plt.savefig("collected_clips_reward_evolution.png") # Save the plot
        plt.close()
    else:
        print("No rewards collected to plot.")
    pbar.close()
    vec_env.close()
    return clips[:num_clips]

def clip_return(c):
    return sum(c["rews"])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pref_ds = PreferenceDataset(device=device) # This will be initialized but not used by the new logic
all_collected_clips_data = []

# Using render_mode=None for potentially faster training if visualization during training is not critical
print("Initializing PPO policy and environment for clip collection task...")
raw_env_ppo = gym.make("Reacher-v4", render_mode=None)
policy = PPO("MlpPolicy", raw_env_ppo,
             verbose=1,
             n_steps=2048, # From original bootstrap
             batch_size=16,  # From original bootstrap
             tensorboard_log="./logs/ppo_reacher_clip_collection/") # New log path

num_collection_intervals = TOTAL_TRAINING_TIMESTEPS_FOR_CLIPS // COLLECTION_INTERVAL_TIMESTEPS

for i in range(num_collection_intervals):
    current_policy_timesteps_start_interval = policy.num_timesteps
    target_timesteps_for_interval_end = (i + 1) * COLLECTION_INTERVAL_TIMESTEPS
    
    print(f"\n--- Interval {i+1}/{num_collection_intervals} ---")
    print(f"Current policy timesteps: {current_policy_timesteps_start_interval}")
    print(f"Training policy for {COLLECTION_INTERVAL_TIMESTEPS} timesteps (target: {target_timesteps_for_interval_end} total)...")
    
    policy.learn(total_timesteps=COLLECTION_INTERVAL_TIMESTEPS,
                 reset_num_timesteps=False if policy.num_timesteps > 0 else True)
    
    print(f"Policy trained. Total timesteps for policy: {policy.num_timesteps}")

    print(f"Collecting {NUM_CLIPS_PER_COLLECTION} clips using the policy trained for {policy.num_timesteps} timesteps...")
    clips = collect_clips(policy, NUM_CLIPS_PER_COLLECTION, SEGMENT_LEN)
    interval_clips_data = []
    for clip_idx, clip_data in enumerate(clips):
        if 'rews' in clip_data and clip_data['rews']:
            reward = clip_return(clip_data)
        else:
            print(f"Warning: Clip {clip_idx} in interval {i} has no rewards. Skipping reward calculation.")
            reward = None
        observations = np.array(clip_data.get('obs', []))
        actions = np.array(clip_data.get('acts', []))
        if observations.size == 0 or actions.size == 0:
            print(f"Warning: Clip {clip_idx} in interval {i} has empty observations or actions. Skipping.")
            continue

        interval_clips_data.append({
            'policy_total_timesteps': policy.num_timesteps,
            'collection_interval_index': i,
            'clip_index_in_interval': clip_idx,
            'observations': observations,
            'actions': actions,
            'reward': reward
        })
    all_collected_clips_data.extend(interval_clips_data)
    print(f"Collected {len(clips)} clips for this interval. Total clips stored so far: {len(all_collected_clips_data)}")

print(f"\nSaving all {len(all_collected_clips_data)} collected clips and rewards to {OUTPUT_FILE_PATH}...")
with open(OUTPUT_FILE_PATH, 'wb') as f:
    pickle.dump(all_collected_clips_data, f)
print(f"Data saved successfully to {OUTPUT_FILE_PATH}")

raw_env_ppo.close()
print("Clip collection task finished.")
exit(0)
