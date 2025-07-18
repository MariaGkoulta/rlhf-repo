from .base_config import *

ENV_ID = "Reacher-v4"
UNHEALTHY_TERMINATION_ENVS = ["Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"]

SEGMENT_LEN = 20 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 20 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 20 # Length of segments to extract from clips
INITIAL_MIN_GAP = 0.25
FINAL_MIN_GAP = 0.25

TOTAL_TARGET_PAIRS = 5000
INITIAL_COLLECTION_FRACTION = 0.3
TOTAL_PPO_TIMESTEPS = 10e6
MAX_EPISODE_STEPS = 50

# PPO hyperparameters
PPO_ENT_COEF = 0.01  # Entropy coefficient for PPO
PPO_LR = 1e-4

# Reward model training hyperparameters
REWARD_MODEL_WEIGHT_DECAY = 1e-2