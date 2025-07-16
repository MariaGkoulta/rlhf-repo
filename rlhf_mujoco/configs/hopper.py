from base_config import *

ENV_ID = "Hopper-v4"

SEGMENT_LEN = 200 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 200 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 200 # Length of segments to extract from clips

USE_BALD = False
# Scales the number of pairs collected per iteration based on the rate.
# If T_cumulative is 0, rate factor is 1, so this is the initial target pairs per iter in main loop.
BASE_PAIRS_PER_ITERATION_SCALER = 50
TOTAL_TARGET_PAIRS = 750
REFERENCE_TIMESTEPS_FOR_RATE = 5e6
TOTAL_PPO_TIMESTEPS = 3e6

# PPO hyperparameters
PPO_TRAINING_PATIENCE = 15

# Reward model training hyperparameters
REWARD_MODEL_DROPOUT_PROB = 0
USE_REWARD_ENSEMBLES = False