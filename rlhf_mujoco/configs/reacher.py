from rlhf_mujoco.configs.base_config import *

ENV_ID = "Reacher-v4"

SEGMENT_LEN = 15 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 15 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 15 # Length of segments to extract from clips
INITIAL_MIN_GAP = 1.5
FINAL_MIN_GAP = 0
# Scales the number of pairs collected per iteration based on the rate.
# If T_cumulative is 0, rate factor is 1, so this is the initial target pairs per iter in main loop.
BASE_PAIRS_PER_ITERATION_SCALER = 50
TOTAL_TARGET_PAIRS = 1000
REFERENCE_TIMESTEPS_FOR_RATE = 5e6
TOTAL_PPO_TIMESTEPS = 10e6
MAX_EPISODE_STEPS = 50

# PPO hyperparameters
PPO_ENT_COEF = 0.03  # Entropy coefficient for PPO
PPO_LR = 1e-4