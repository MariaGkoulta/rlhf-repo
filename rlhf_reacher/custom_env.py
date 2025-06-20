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

        # Variables for running normalization of raw learned rewards
        self.raw_learned_r_count = 0
        self.raw_learned_r_mean = 0.0
        self.raw_learned_r_M2 = 0.0 # M2 is the sum of squares of differences from the current mean
        self.reward_norm_epsilon = 1e-8 # Epsilon for reward normalization stddev
        self.normalize_rewards = normalize_rewards # Flag to enable/disable reward normalization

    def get_and_clear_episode_rewards(self):
        """Returns the collected episode rewards and clears the internal lists."""
        true = self.ep_true
        learned = self.ep_learned
        self.ep_true = []
        self.ep_learned = []
        return true, learned

    def _update_running_stats(self, raw_value: float): # Takes raw value from reward model
        self.raw_learned_r_count += 1
        delta = raw_value - self.raw_learned_r_mean
        self.raw_learned_r_mean += delta / self.raw_learned_r_count
        delta2 = raw_value - self.raw_learned_r_mean 
        self.raw_learned_r_M2 += delta * delta2
    
    def _get_running_std(self) -> float:
        """Calculates running standard deviation."""
        if self.raw_learned_r_count < 2:
            return 0.0 
        variance = self.raw_learned_r_M2 / self.raw_learned_r_count # This is population variance if count is N, sample if N-1
        return np.sqrt(variance)

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
        
        reward_for_ppo = learned_r_raw 

        if self.normalize_rewards:
            self._update_running_stats(learned_r_raw)
            current_std = self._get_running_std()
            if current_std > self.reward_norm_epsilon:
                normalized_learned_r = (learned_r_raw - self.raw_learned_r_mean) / current_std
            else:
                normalized_learned_r = learned_r_raw - self.raw_learned_r_mean
            reward_for_ppo = normalized_learned_r
        
        self._learned_sum += reward_for_ppo 

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
        return obs, reward_for_ppo, terminated, truncated, info