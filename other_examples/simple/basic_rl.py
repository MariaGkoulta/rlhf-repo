import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment without rendering during training
train_env = gym.make('CartPole-v1')

# Train model
model = PPO('MlpPolicy', train_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_cartpole")

# Evaluate model
mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Create a separate environment for visualization
eval_env = gym.make('CartPole-v1', render_mode="human")
obs, info = eval_env.reset()
done = False
truncated = False

# Visualization loop
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = eval_env.step(action)

eval_env.close()
train_env.close()