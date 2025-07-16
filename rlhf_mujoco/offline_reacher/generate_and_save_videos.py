import os
import sys
import gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO 
import datetime

policy_path = sys.argv[1]
policy = PPO.load(policy_path) # Load the policy
print("Evaluating final policy and recording videos...")

num_videos = 5
current_time = datetime.datetime.now()
time = current_time.strftime("%Y%m%d_%H%M%S")
base_video_folder = f"videos/policy_evaluation_{time}_" + os.path.splitext(os.path.basename(policy_path))[0]

for i in range(num_videos):
    video_folder = os.path.join(base_video_folder, f"video_{i}")
    os.makedirs(video_folder, exist_ok=True)
    eval_env = gym.make("Reacher-v4", render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        video_folder=video_folder,
        episode_trigger=lambda ep_id: ep_id == 0,
        name_prefix=f"final-reacher-eval-{i}"
    )
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
    eval_env.close()
    print(f"Video {i} saved to {video_folder}/")

print(f"All {num_videos} policy evaluation videos saved to {base_video_folder}/")