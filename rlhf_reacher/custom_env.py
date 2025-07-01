import gymnasium as gym
import numpy as np
from collections import deque

class LearnedRewardEnv(gym.Wrapper):
    def __init__(self, env, reward_model, normalize_rewards=True, window_size=5000):
        super().__init__(env)
        self.reward_model = reward_model
        self.prev_obs = None 
        self._true_sum = 0.0
        self._learned_sum = 0.0 
        self.ep_true = []
        self.ep_learned = []

        self.reward_norm_epsilon = 1e-8 # Epsilon for reward normalization stddev
        self.normalize_rewards = normalize_rewards # Flag to enable/disable reward normalization
        if self.normalize_rewards:
            self.raw_reward_buffer = deque(maxlen=window_size)

    def get_and_clear_episode_rewards(self):
        """Returns the collected episode rewards and clears the internal lists."""
        true = self.ep_true
        learned = self.ep_learned
        self.ep_true = []
        self.ep_learned = []
        return true, learned

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_obs = obs
        self._true_sum = 0.0
        self._learned_sum = 0.0
        return obs, info

    def step(self, action):
        obs, true_r, terminated, truncated, info = super().step(action)

        learned_r_raw = self.reward_model.predict_reward(self.prev_obs, action)
        
        if isinstance(learned_r_raw, tuple):
            learned_r_raw = learned_r_raw[0]

        self._true_sum += true_r
        
        reward_for_ppo = learned_r_raw 
        if self.normalize_rewards:
            self.raw_reward_buffer.append(learned_r_raw)
            if len(self.raw_reward_buffer) > 1:
                mean = np.mean(self.raw_reward_buffer)
                std = np.std(self.raw_reward_buffer)
                reward_for_ppo = (learned_r_raw - mean) / (std + self.reward_norm_epsilon)
        self._learned_sum += reward_for_ppo 

        done = terminated or truncated
        
        if done:
            info = info.copy()
            info["episode"] = {
                "r_true": self._true_sum,
                "r_learned": self._learned_sum,
                'r_per_step': learned_r_raw
            }
            self.ep_true.append(self._true_sum)
            self.ep_learned.append(self._learned_sum)

        self.prev_obs = obs
        return obs, reward_for_ppo, terminated, truncated, info