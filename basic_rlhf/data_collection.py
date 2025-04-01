import gymnasium as gym
import numpy as np
import pickle
import torch
from policy_network import PolicyNetwork

def train_policy_baseline(env, num_rollouts, gamma=0.99, lr=1e-2, device='cpu'):
    policy = PolicyNetwork(state_dim=env.observation_space.shape[0]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for i in range(num_rollouts):
        state, _ = env.reset()
        done = False
        rewards = []
        log_probs = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            probs = policy(state_tensor)
            action = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[0, action])
            next_state, reward, done, truncated, info = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
        
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            discounted_rewards.insert(0,R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        loss = sum(-log_prob * R for log_prob, R in zip(log_probs, discounted_rewards))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Baseline training: Episode {i+1}/{i}, Loss: {loss.item():.4f}")

    return policy

def run_policy(env, policy, max_steps=200):
    state, _ = env.reset()
    trajectory = []
    total_reward = 0

    for _ in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(state_tensor)
        action = torch.multinomial(probs, 1).item()
        next_state, reward, done, truncated, info = env.step(action)
        trajectory.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state
        if done or truncated:
            break
    return trajectory, total_reward


def collect_data(num_policies=5, num_trajectories=100, device='cpu'):
    env = gym.make('CartPole-v1')
    trajectories = {}
    policies = []
    for i in range(num_policies):
        traj_list = []
        num_rollouts = (i+1) * 20
        policy = train_policy_baseline(env, num_rollouts)
        policies.append(policy)
        for _ in range(num_trajectories):
            traj, rewards = run_policy(env, policy)
            traj_list.append((traj, rewards))
        trajectories[f"Level_{i}"] = traj_list
    
    env.close()
    return trajectories, policies

if __name__ == "__main__":
    trajectories, policies = collect_data()
    with open('trajectories.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
        print(f"Data collection complete. Saved {len(trajectories)} trajectories.")
    for level, trajs in trajectories.items():
        avg_reward = np.mean([tr[1] for tr in trajs])
        print(f"{level}: {len(trajs)} trajectories, average reward: {avg_reward:.2f}")
