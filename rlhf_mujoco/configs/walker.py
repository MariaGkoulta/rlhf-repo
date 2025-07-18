from .base_config import *

ENV_ID = "Walker2d-v5"

SEGMENT_LEN = 250
INITIAL_SEGMENT_LEN = 250
FINAL_SEGMENT_LEN = 250

TOTAL_TARGET_PAIRS = 1000
TOTAL_PPO_TIMESTEPS = 3e6

# Reward model training hyperparameters
USE_REWARD_ENSEMBLES = False