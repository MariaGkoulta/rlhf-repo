from .base_config import *

ENV_ID = "Hopper-v4"

SEGMENT_LEN = 200 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 200 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 200 # Length of segments to extract from clips

USE_BALD = False
TOTAL_TARGET_PAIRS = 1000
TOTAL_PPO_TIMESTEPS = 3e6

# PPO hyperparameters
PPO_TRAINING_PATIENCE = 15

# Reward model training hyperparameters
REWARD_MODEL_DROPOUT_PROB = 0
USE_REWARD_ENSEMBLES = False

FEEDBACK_TYPE = "evaluative"