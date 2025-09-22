from .base_config import *

ENV_ID = "Swimmer-v5"

SEGMENT_LEN = 50 # Length of segments to extract from clips
INITIAL_SEGMENT_LEN = 50 # Length of segments to extract from clips
FINAL_SEGMENT_LEN = 50 # Length of segments to extract from clips
############################################################
# Swimmer fine‑tuning adjustments
# Rationale:
# - Swimmer underperformed vs Hopper/Walker/Cheetah.
# - Provide longer initial policy warmup so preference/evaluative data
#   are collected from a semi‑competent policy (stabilizes reward model).
# - Slightly lower total target feedback and use higher initial fraction
#   to bootstrap faster.
# - Give PPO more exploration (non‑zero entropy) & slightly higher LR.
# - Adjust evaluative rating range to expected Swimmer return scale (0~150)
#   so model targets a meaningful dynamic range. (Original 0..50 was tight.)
# - Reduce min gap to allow more diverse early pairs when returns small.
############################################################

# Feedback collection targets
TOTAL_TARGET_PAIRS = 1000          # was 1500 – reduce labeling burden
INITIAL_COLLECTION_FRACTION = 0.4  # was 0.3 – gather more seed signal

# PPO / training horizon
TOTAL_PPO_TIMESTEPS = 3e6          # slight reduction (faster iteration) vs 4e6
INITIAL_POLICY_TS = 75_000         # warmup (base default was 1) – critical

# PPO hyperparameters (override base)
PPO_ENT_COEF = 0.01                # was 0.0 – encourage exploration
PPO_LR = 2.5e-4                    # slight bump vs base 3e-4 acceptable
PPO_N_STEPS = 2048                 # keep default but explicit for clarity
PPO_BATCH_SIZE = 64                # inherit but restated
PPO_N_EPOCHS = 10                  # inherit but restated

# Reward model adjustments
REWARD_MODEL_DROPOUT_PROB = 0.1    # keep modest dropout
REWARD_MODEL_EPOCHS = 120          # allow a bit more fitting if early noise
REWARD_MODEL_PATIENCE = 12         # slightly longer patience

# Evaluative feedback configuration
FEEDBACK_TYPE = "evaluative"       # keep evaluative (easier than prefs here)
EVALUATIVE_RATING_RANGE = (0, 150) # widen range to match environment returns
EVALUATIVE_RATING_SCALE = None     # use raw scale – normalization handled upstream
USE_PROBABILISTIC_MODEL = True     # keep probabilistic for BALD extensibility later

# Gap / segment specifics (loosen min gap early to get more usable pairs)
INITIAL_MIN_GAP = 0.1
FINAL_MIN_GAP = 0.1