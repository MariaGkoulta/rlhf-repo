
ENV_ID = "Reacher-v4"
# For some MuJoCo envs, we might want to disable termination on unhealthy states
# to learn from a wider range of behaviors.
# Add environment IDs to this list to apply the setting.
TERMINATE_WHEN_UNHEALTHY = False
UNHEALTHY_TERMINATION_ENVS = ["Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"]

INITIAL_POLICY_TS = 1
NORMALIZE_REWARDS = True
EXTRACT_SEGMENTS = True  # If True, segments are extracted from clips
SEGMENT_LEN = 15 # Length of segments to extract from clips
NUM_EPISODES_TO_COLLECT_INITIAL = 200 # Number of full episodes to collect initially.
NUM_EPISODES_TO_COLLECT_PER_UPDATE = 200  # Number of full episodes to collect in each iteration.
# Only used if EXTRACT_SEGMENTS is True:
TARGET_NUM_SEGMENTS_IF_EXTRACTING_INITIAL = 1000 # Target number of segments to sample from initial episodes.
TARGET_NUM_SEGMENTS_IF_EXTRACTING_PER_UPDATE = 200
INITIAL_MIN_GAP = 1.5
FINAL_MIN_GAP = 0
NUM_BINS = 120
BALD_POOL_SIZE = 50000
BALD_K = 10000
BALD_T = 10
# Scales the number of pairs collected per iteration based on the rate.
# If T_cumulative is 0, rate factor is 1, so this is the initial target pairs per iter in main loop.
BASE_PAIRS_PER_ITERATION_SCALER = 50
TOTAL_TARGET_PAIRS = 7000
INITIAL_COLLECTION_FRACTION = 0.3
PPO_TIMESTEPS_PER_ITER = 20000  # Train policy more often with fewer steps
REFERENCE_TIMESTEPS_FOR_RATE = 5e6
TOTAL_PPO_TIMESTEPS = 10e6
MAX_EPISODE_STEPS = 50

# PPO hyperparameters
PPO_ENT_COEF = 0.03  # Entropy coefficient for PPO
PPO_LR = 1e-4
PPO_N_EPOCHS = 4
PPO_BATCH_SIZE = 64
PPO_N_STEPS = 2048

# Reward model training hyperparameters
REWARD_MODEL_LEARNING_RATE = 1e-3
REWARD_MODEL_WEIGHT_DECAY = 1e-2
REWARD_MODEL_REGULARIZATION_WEIGHT = 1e-5
REWARD_MODEL_EPOCHS = 100
REWARD_MODEL_PATIENCE = 10