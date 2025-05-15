import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from scipy.stats import pearsonr
from reward_model import RewardModel


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

def make_wrapped_env(reward_model_to_wrap, render_mode="rgb_array", env_id="Reacher-v4"):
    env_core = gym.make(env_id, render_mode=render_mode)
    env_learned_reward = LearnedRewardEnv(env_core, reward_model_to_wrap)
    env_monitored = Monitor(env_learned_reward)
    return env_monitored


raw_env = gym.make("Reacher-v4", render_mode="rgb_array")
policy  = PPO("MlpPolicy", raw_env, 
              verbose=1, n_steps=2048, batch_size=16, tensorboard_log="./logs/ppo_reacher/")
policy.learn(total_timesteps=50_000)
policy.save("ppo_reacher_initial")
raw_env.close()

temp_env = gym.make("Reacher-v4")
obs_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.shape[0]
temp_env.close()

loaded_reward_model = RewardModel(obs_dim, action_dim)
reward_model_path = 'trained_reward_model_1505.pth'
try:
    state_dict = torch.load(reward_model_path, map_location=torch.device('cpu')) # Load to CPU first
    loaded_reward_model.load_state_dict(state_dict)
    print(f"Successfully loaded reward model state_dict from {reward_model_path}")
except Exception as e:
    print(f"Error loading reward model from {reward_model_path}: {e}")
    print("Please ensure 'trained_reward_model.pth' exists and contains a compatible state_dict for the RewardModel class.")
    exit()

loaded_reward_model.eval() # Set to evaluation mode

callback = TrueRewardCallback()
train_env = make_wrapped_env(loaded_reward_model, render_mode="rgb_array") # or render_mode=None for faster training


# --- Define and Train the PPO Agent ---
# model = PPO(
#     policy="MlpPolicy",
#     env=train_env,
#     n_steps=2048,
#     batch_size=64,
#     gae_lambda=0.95,
#     gamma=0.99,
#     ent_coef=0.0,
#     learning_rate=3e-4,
#     clip_range=0.2,
#     verbose=1,
#     tensorboard_log="./ppo_reacher_tensorboard/" # Optional: for logging
# )

print("Starting training with learned reward model...")
policy.set_env(train_env)
policy.learn(total_timesteps=1_000_000, callback=callback) # Adjust timesteps as needed
policy.save("ppo_reacher_learned_reward")
print("Training finished and model saved as ppo_reacher_learned_reward.")
train_env.close()

# --- Create a new environment for final policy video evaluation (using TRUE environment rewards) ---
print("Evaluating final policy (on true environment) and recording video...")
video_output_folder = "videos/final_policy_evaluation_learned_reward_agent"
os.makedirs(video_output_folder, exist_ok=True)

eval_env = gym.make("Reacher-v4", render_mode="rgb_array")
eval_env = RecordVideo(eval_env,
                       video_folder=video_output_folder,
                       episode_trigger=lambda ep_id: ep_id == 0, # Record only the first episode
                       name_prefix="final-reacher-offline-eval-learned-agent")

obs, _ = eval_env.reset()
# Reacher-v4 has max_episode_steps=50. 200 steps = 4 episodes.
for _ in range(200): # Run for a few episodes
    action, _states = policy.predict(obs, deterministic=True) # Use the trained model
    obs, rewards, dones, truncated, info = eval_env.step(action)
    if dones or truncated:
        obs, _ = eval_env.reset()

eval_env.close()
print(f"Final policy evaluation video saved in {video_output_folder}.")