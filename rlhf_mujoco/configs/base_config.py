# For some MuJoCo envs, we might want to disable termination on unhealthy states
# to learn from a wider range of behaviors.
# Add environment IDs to this list to apply the setting.
TERMINATE_WHEN_UNHEALTHY = False
UNHEALTHY_TERMINATION_ENVS = ["Hopper-v4", "Walker2d-v5", "Ant-v4", "Humanoid-v4"]

INITIAL_POLICY_TS = 1
NORMALIZE_REWARDS = True
EXTRACT_SEGMENTS = True  # If True, segments are extracted from clips

NUM_EPISODES_TO_COLLECT_INITIAL = 200 # Number of full episodes to collect initially.
NUM_EPISODES_TO_COLLECT_PER_UPDATE = 200  # Number of full episodes to collect in each iteration.
# Only used if EXTRACT_SEGMENTS is True:
TARGET_NUM_SEGMENTS_IF_EXTRACTING_INITIAL = 1000 # Target number of segments to sample from initial episodes.
TARGET_NUM_SEGMENTS_IF_EXTRACTING_PER_UPDATE = 200
INITIAL_MIN_GAP = 0.25
FINAL_MIN_GAP = 0.25
NUM_BINS = 120
BALD_POOL_SIZE = 50000
BALD_K = 10000
BALD_T = 20
INITIAL_COLLECTION_FRACTION = 0.1
MAX_EPISODE_STEPS = 1000

# PPO hyperparameters
PPO_ENT_COEF = 0.01  # Entropy coefficient for PPO
PPO_LR = 3e-4
PPO_N_EPOCHS = 10
PPO_BATCH_SIZE = 64
PPO_N_STEPS = 2048
PPO_TIMESTEPS_PER_ITER = 20000  # Train policy more often with fewer steps

# Reward model training hyperparameters
REWARD_MODEL_LEARNING_RATE = 1e-3
REWARD_MODEL_WEIGHT_DECAY = 1e-3
REWARD_MODEL_REGULARIZATION_WEIGHT = 1e-5
REWARD_MODEL_EPOCHS = 100
REWARD_MODEL_PATIENCE = 10
REWARD_MODEL_DROPOUT_PROB = 0.1
BALD_REWARD_MODEL_DROPOUT_PROB = 0.4
REWARD_ENSEMBLES = 5

# Feedback type configuration
FEEDBACK_TYPE = "preference"  # Options: "preference", "evaluative"
EVALUATIVE_RATING_BINS = 10  # Number of rating bins for evaluative feedback (1-10)
DISCOUNT_FACTOR = 0.99  # Gamma for discounted return calculation
EVALUATIVE_RATING_RANGE = (0, 10)  # Range of ratings for evaluative feedback
EVALUATIVE_RATING_SCALE = 10