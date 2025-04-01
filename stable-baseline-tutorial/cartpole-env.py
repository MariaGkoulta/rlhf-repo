import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


# Wrapper for the CartPole environment that keeps track of the rewards
class PreferenceCartPoleEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_data = []

    def reset(self, **kwargs):
        self.episode_data = []
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        next_state, true_reward, terminated, truncated, info = self.env.step(action)
        self.episode_data.append((next_state, true_reward))
        info['true_reward'] = true_reward
        return next_state, true_reward, terminated, truncated, info

    
def simulate_preferences(traj1, traj2, noise_std=0.1):
    R1 = sum(r for _, r in traj1)
    R2 = sum(r for _, r in traj2)

    # Add Gaussian noise to the rewards
    R1 += np.random.normal(0, noise_std)
    R2 += np.random.normal(0, noise_std)

    return 1 if R1 > R2 else 0
    

# Simple reward model that predicts the reward based on the state
class RewardModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    
def pairwise_loss(r1, r2, label):
    if label == 1:
        return -torch.log(torch.sigmoid(r1 - r2) + 1e-8)
    else:
        return -torch.log(torch.sigmoid(r2 - r1) + 1e-8)
    
episode_buffer = []

# Update the reward model using a buffer of episodes
def update_reward_model(buffer, reward_model, optimizer_rm, update_steps=50):
    for _ in range(update_steps):
        if len(buffer) < 2:
            continue

        idx1, idx2 = np.random.choice(len(buffer), size=2, replace=False)
        ep1 = buffer[idx1]
        ep2 = buffer[idx2]
        _, _, traj1 = ep1
        _, _, traj2 = ep2
        label = simulate_preferences(traj1, traj2, noise_std=0.1)
        
        states1 = torch.tensor(np.array([s for s, _ in traj1]), dtype=torch.float32)
        states2 = torch.tensor(np.array([s for s, _ in traj2]), dtype=torch.float32)
        preds1 = reward_model(states1).sum() 
        preds2 = reward_model(states2).sum()
    
        loss = pairwise_loss(preds1, preds2, label)
        optimizer_rm.zero_grad()
        loss.backward()
        optimizer_rm.step()

# Create wrapper for reward model
class RLHFRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model
        self.state_history = []
        self.prev_cumulative_reward = 0.0
    
    def reset(self, **kwargs):
        self.state_history = []
        self.prev_cumulative_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        next_state, true_reward, terminated, truncated, info = self.env.step(action)
        
        # Add new state to history
        self.state_history.append(next_state)
        
        # Calculate cumulative reward for entire trajectory so far
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(self.state_history))
            current_cumulative_reward = self.reward_model(states_tensor).sum().item()
            
            # Calculate incremental reward (difference from previous cumulative)
            incremental_reward = current_cumulative_reward - self.prev_cumulative_reward
            self.prev_cumulative_reward = current_cumulative_reward
        
        # Store true reward in info
        info['true_reward'] = true_reward
        
        return next_state, incremental_reward, terminated, truncated, info
    
# Initialize components
env_true = PreferenceCartPoleEnv(gym.make("CartPole-v1"))
model = PPO("MlpPolicy", env_true, verbose=1)
input_dim = env_true.observation_space.shape[0]
reward_model = RewardModel(input_dim)
optimizer_rm = optim.Adam(reward_model.parameters(), lr=0.001)

episode_numbers = []
true_rewards_history = []
predicted_rewards_history = []

num_episodes = 20
buffer_update_interval = 10  
policy_update_timesteps = 1000

# Training loop
for episode in range(num_episodes):
    state, _ = env_true.reset()
    done = False
    ep_reward = 0
    states_in_episode = []
    
    while not done:
        action, _ = model.predict(state)
        state, true_reward, terminated, truncated, info = env_true.step(action)
        states_in_episode.append(state)  # Store all states visited in this episode
        done = terminated or truncated
        ep_reward += true_reward
    episode_buffer.append((state, ep_reward, env_true.episode_data.copy()))

    with torch.no_grad():
        episode_states = torch.FloatTensor(np.array(states_in_episode))
        episode_pred_rewards = reward_model(episode_states).sum().item()
        
    # Record performance metrics
    episode_numbers.append(episode)
    true_rewards_history.append(ep_reward)
    predicted_rewards_history.append(episode_pred_rewards)


    if (episode + 1) % buffer_update_interval == 0:
        update_reward_model(episode_buffer, reward_model, optimizer_rm, update_steps=50)
        print(f"Updated reward model at episode {episode+1}")
    
    # At a midpoint, switch the environment to use the learned reward.
    if episode == 4:
        env_rlhf = RLHFRewardWrapper(PreferenceCartPoleEnv(gym.make("CartPole-v1")), reward_model)
        model.set_env(env_rlhf)
        print("Switched to RLHF reward for policy training.")

    # Perform policy update using PPO.
    model.learn(total_timesteps=policy_update_timesteps, reset_num_timesteps=False)
    print(f"Episode {episode+1} true cumulative reward: {ep_reward:.2f}")

# PLot the results
plt.figure(figsize=(8, 4))
plt.plot(episode_numbers, true_rewards_history, label="True Cumulative Reward", marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("True Cumulative Reward")
plt.title("Agent Performance Over Episodes")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(true_rewards_history, predicted_rewards_history, alpha=0.7)
min_val = min(min(true_rewards_history), min(predicted_rewards_history))
max_val = max(max(true_rewards_history), max(predicted_rewards_history))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal: y=x")
plt.xlabel("True Cumulative Reward")
plt.ylabel("Predicted Reward")
plt.title("Reward Model Accuracy")
plt.legend()
plt.grid(True)
plt.show()