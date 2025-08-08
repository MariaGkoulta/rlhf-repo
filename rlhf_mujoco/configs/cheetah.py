from .base_config import *

ENV_ID = "HalfCheetah-v5"

SEGMENT_LEN = 40 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 40 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 40 # Length of segments to extract from clips

TOTAL_TARGET_PAIRS = 500
TOTAL_PPO_TIMESTEPS = 5e6

FEEDBACK_TYPE = "preference" # Type of feedback to use
EVALUATIVE_RATING_RANGE=(-30, 120)
EVALUATIVE_RATING_SCALE=None
USE_PROBABILISTIC_MODEL = False  
INITIAL_COLLECTION_FRACTION = 0.1