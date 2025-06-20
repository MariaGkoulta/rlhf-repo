import gymnasium as gym
import numpy as np

class LearnedRewardEnv(gym.Wrapper):
    def __init__(self, env, reward_model, normalize_rewards=True):
        super().__init__(env)
        self.reward_model = reward_model
        self.prev_obs = None 
        self._true_sum = 0.0
        self._learned_sum = 0.0 
        self.ep_true = []
        self.ep_learned = []

        # Note: normalize_rewards parameter is kept for compatibility but not used
        # since VecNormalize handles reward normalization

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

        self._true_sum += true_r
        self._learned_sum += learned_r_raw

        done = terminated or truncated
        
        if done:
            info = info.copy()
            info["episode"] = {
                "r_true": self._true_sum,
                "r_learned": self._learned_sum
            }
            self.ep_true.append(self._true_sum)
            self.ep_learned.append(self._learned_sum)

        self.prev_obs = obs
        return obs, learned_r_raw, terminated, truncated, info