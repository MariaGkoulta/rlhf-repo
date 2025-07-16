import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo


env = gym.make("Reacher-v4", render_mode="rgb_array")        
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.0,
    learning_rate=3e-4,
    clip_range=0.2,
    verbose=1,
)
model.learn(total_timesteps=1_000_000)
model.save("ppo_reacher")
env.close()

# --- Create a new environment for final policy video evaluation ---
print("Evaluating final policy and recording video...")
eval_env = gym.make("Reacher-v4", render_mode="rgb_array")
eval_env = RecordVideo(eval_env,
                       video_folder="videos/final_policy_evaluation", 
                       episode_trigger=lambda ep_id: ep_id == 0,
                       name_prefix="final-reacher-eval")

obs, _ = eval_env.reset()
for _ in range(200 * 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, truncated, info = eval_env.step(action)
    if dones or truncated:
        obs, _ = eval_env.reset()

eval_env.close()
print("Final policy evaluation video saved.")