import inspect
import gymnasium as gym
import numpy as np
from scipy.stats import pearsonr
from stable_baselines3.common.callbacks import BaseCallback

class NoSeedArgumentWrapper(gym.Wrapper):
    """
    A wrapper that calls the underlying environment's reset method
    without the 'seed' keyword argument, as the underlying environment's
    reset signature might not support it. It passes 'options' if the
    underlying environment's reset supports it.
    This wrapper should be applied directly to the environment created by
    gym.make() before other wrappers like TimeLimit.
    """
    def __init__(self, env):
        super().__init__(env)
        self._env_reset_accepts_options = False
        try:
            # Check the signature of the reset method of the wrapped environment (self.env)
            sig = inspect.signature(self.env.reset)
            if 'options' in sig.parameters:
                self._env_reset_accepts_options = True
        except (TypeError, ValueError):
            # inspect.signature can fail on some callables, or env might not have reset
            # Default to not passing options if introspection fails.
            pass

    def reset(self, *, seed=None, options=None):
        # This method is called by an outer wrapper (e.g., TimeLimit) or SB3,
        # which passes 'seed' and 'options'.
        # We call self.env.reset (the base environment's reset) *without* 'seed'.
        if self._env_reset_accepts_options:
            # If the base environment's reset takes 'options', pass it through.
            # 'options' itself could be None, which is fine by Gymnasium spec.
            return self.env.reset(options=options)
        else:
            return self.env.reset()
        
class TrueRewardCallback(BaseCallback):

    def __init__(self, verbose=0, patience=15):
        super().__init__(verbose)
        self.patience = patience
        self.best_mean_reward = -float('inf')
        self.patience_counter = 0
        self._stop_training = False

    def _on_training_start(self) -> None:
        """
        This method is called at the beginning of a `learn()` call.
        """
        self.patience_counter = 0
        self._stop_training = False

    def _on_rollout_end(self) -> None:
        true_returns = []
        learned_returns = []
        for ep_list in self.training_env.get_attr("ep_true"):
            true_returns.extend(ep_list)
        for ep_list in self.training_env.get_attr("ep_learned"):
            learned_returns.extend(ep_list)
        if true_returns:
            current_mean_reward = np.mean(true_returns)
            self.logger.record("rollout/ep_true_mean", current_mean_reward)
            self.logger.record("rollout/ep_true_variance", np.var(true_returns))  # Log variance of true rewards

            if current_mean_reward > self.best_mean_reward:
                self.best_mean_reward = current_mean_reward
                self.patience_counter = 0
                if self.verbose > 0:
                    print(f"New best mean reward: {self.best_mean_reward:.2f}. Resetting patience.")
            else:
                self.patience_counter += 1
                if self.verbose > 0:
                    print(f"No improvement in mean reward for {self.patience_counter} rollout(s).")
            
            if self.patience_counter >= self.patience:
                self._stop_training = True
                if self.verbose > 0:
                    print(f"Stopping training early: patience of {self.patience} rollouts reached.")

        if learned_returns:
            self.logger.record("rollout/ep_learned_mean", np.mean(learned_returns))
            self.logger.record("rollout/ep_learned_variance", np.var(learned_returns))  # Log variance of learned rewards
        if len(true_returns) == len(learned_returns) > 1:
            r, p = pearsonr(true_returns, learned_returns)
            self.logger.record("rollout/pearson_r_true_vs_learned", r)
            self.logger.record("rollout/pearson_p_value", p)
        else:
            self.logger.record("rollout/pearson_r_true_vs_learned", np.nan)
            self.logger.record("rollout/pearson_p_value", np.nan)

        # for subenv in self.training_env.envs:
        #     if hasattr(subenv, "ep_true"):
        #         subenv.ep_true.clear()
        #         subenv.ep_learned.clear()

    def _on_step(self) -> bool:
        if self._stop_training:
            return False
        return True