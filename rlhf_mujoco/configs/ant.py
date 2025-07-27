from .base_config import *

ENV_ID = "Ant-v5"

SEGMENT_LEN = 30 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 30 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 30 # Length of segments to extract from clips

TOTAL_TARGET_PAIRS = 4000
TOTAL_PPO_TIMESTEPS = 5e6