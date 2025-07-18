from .base_config import *

ENV_ID = "Swimmer-v5"

SEGMENT_LEN = 200 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 200 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 200 # Length of segments to extract from clips

TOTAL_TARGET_PAIRS = 1000
TOTAL_PPO_TIMESTEPS = 3e6

# PPO hyperparameters
PPO_ENT_COEF = 0.0  # Entropy coefficient for PPO

# Reward model training hyperparameters
USE_REWARD_ENSEMBLES = False