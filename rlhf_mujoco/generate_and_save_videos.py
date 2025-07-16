import os
import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO 
import datetime

policy_path = sys.argv[1]
policy = PPO.load(policy_path) # Load the policy
print("Evaluating final policy and recording videos...")

# --- Infer environment name from policy path ---
policy_filename_no_ext = os.path.splitext(os.path.basename(policy_path))[0]
# Assuming policy name format like "ppo_ENVNAME_..." or "ENVNAME_..."
if policy_filename_no_ext.startswith("ppo_"):
    env_name_inferred = policy_filename_no_ext.split('_')[1]
else:
    # Fallback or adjust as needed if the policy name doesn't start with "ppo_"
    # This assumes the environment name is the first part if "ppo_" is not present.
    env_name_inferred = policy_filename_no_ext.split('_')[0]

env_id = "HalfCheetah-v4"  # Default value, will be overwritten if inference is successful
# env_id = f"{env_name_inferred.capitalize()}-v4" # e.g., "Reacher-v4" or "Swimmer-v4"
print(f"Inferred environment ID: {env_id}")
# --- End of inference ---

num_videos = 5
current_time = datetime.datetime.now()
time = current_time.strftime("%Y%m%d_%H%M%S")
base_video_folder = f"videos/policy_evaluation_{time}_" + policy_filename_no_ext # Use policy_filename_no_ext for clarity

for i in range(num_videos):
    video_folder = os.path.join(base_video_folder, f"video_{i}")
    os.makedirs(video_folder, exist_ok=True)
    eval_env = gym.make(env_id, render_mode="rgb_array") # Use inferred env_id
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_folder,
        episode_trigger=lambda ep_id: ep_id == 0,
        name_prefix=f"final-{env_name_inferred.lower()}-eval-{i}" # Use inferred env_name
    )
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
    eval_env.close()
    print(f"Video {i} saved to {video_folder}/")

print(f"All {num_videos} policy evaluation videos saved to {base_video_folder}/")