from base_config import *

ENV_ID = "HalfCheetah-v5"

SEGMENT_LEN = 40 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 40 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 40 # Length of segments to extract from clips

# Scales the number of pairs collected per iteration based on the rate.
# If T_cumulative is 0, rate factor is 1, so this is the initial target pairs per iter in main loop.
TOTAL_TARGET_PAIRS = 500
TOTAL_PPO_TIMESTEPS = 5e6