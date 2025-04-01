import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from policy_network import PolicyNetwork
from reward_model import RewardModel
import matplotlib.pyplot as plt
import numpy as np


def select_action(policy, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = policy(state_tensor)
    action = torch.multinomial(probs, 1).item()
    log_prob = torch.log(probs[0, action])
    return action, log_prob

def train_rlhf_policy(episodes=200, gamma=0.99, learning_rate=1e-3, device='cpu'):
    env = gym.make('CartPole-v1')
    policy = PolicyNetwork(state_dim=4).to(device)
    reward_model = RewardModel(state_dim=4).to(device)
    reward_model.load_state_dict(torch.load('reward_model.pth', map_location=device))
    reward_model.eval()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    episode_true_rewards = []
    episode_predicted_rewards = []
    episode_lengths = []


    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        predicted_rewards = []
        true_rewards = []
        done = False
        steps = 0
        while not done:
            action, log_prob = select_action(policy, state)
            next_state, true_reward, done, truncated, info = env.step(action)
            log_probs.append(log_prob)
            true_rewards.append(true_reward)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                r_pred = reward_model(state_tensor)
            predicted_rewards.append(r_pred.item())
            state = next_state
            steps += 1
        
        episode_lengths.append(steps)
        episode_true_rewards.append(sum(true_rewards))
        episode_predicted_rewards.append(sum(predicted_rewards))

        
        discounted_returns = []
        R = 0
        for r in reversed(predicted_rewards):
            R = r + gamma * R
            discounted_returns.insert(0, R)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32, device=device)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)
        
        policy_loss = sum(-log_prob * R for log_prob, R in zip(log_probs, discounted_returns))
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if (episode+1) % 10 == 0:
            print(f"RLHF Policy Training Episode {episode+1}/{episodes}, Loss: {policy_loss.item():.4f}")
    
    env.close()
    torch.save(policy.state_dict(), 'rlhf_policy.pth')
    print("RLHF Policy saved as 'rlhf_policy.pth'.")

     # Plot reward comparison
    plot_rewards(episode_true_rewards, episode_predicted_rewards, episode_lengths)
    
    return policy, episode_true_rewards, episode_predicted_rewards


def plot_rewards(true_returns, pred_returns, episode_lengths):
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Episode Returns
    plt.subplot(2, 2, 1)
    episodes = list(range(1, len(true_returns) + 1))
    
    plt.plot(episodes, true_returns, 'b-', label='True Returns')
    plt.plot(episodes, pred_returns, 'r-', label='Predicted Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode Returns: True vs Predicted')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Correlation
    plt.subplot(2, 2, 2)
    # Scale predicted returns to match true returns for better visualization
    scale_factor = np.mean(true_returns) / (np.mean(pred_returns) if np.mean(pred_returns) != 0 else 1)
    scaled_pred_returns = np.array(pred_returns) * scale_factor
    
    plt.scatter(true_returns, scaled_pred_returns, alpha=0.6)
    max_val = max(max(true_returns), max(scaled_pred_returns))
    min_val = min(min(true_returns), min(scaled_pred_returns))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('True Returns')
    plt.ylabel('Scaled Predicted Returns')
    plt.title('Correlation: True vs Predicted Returns')
    plt.grid(True)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(true_returns, pred_returns)[0, 1]
    plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Plot 3: Running average
    plt.subplot(2, 2, 3)
    window_size = min(20, len(true_returns))
    true_avg = [np.mean(true_returns[max(0, i-window_size):i+1]) for i in range(len(true_returns))]
    pred_avg = [np.mean(pred_returns[max(0, i-window_size):i+1]) for i in range(len(pred_returns))]
    
    plt.plot(episodes, true_avg, 'b-', label=f'True Returns ({window_size}-ep avg)')
    plt.plot(episodes, pred_avg, 'r-', label=f'Predicted Returns ({window_size}-ep avg)')
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title(f'Running Average Returns ({window_size}-episode window)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Learning curve
    plt.subplot(2, 2, 4)
    plt.plot(episodes, episode_lengths, 'g-')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length (Learning Curve)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reward_comparison.png')
    plt.show()
    print("Plot saved as 'reward_comparison.png'")

if __name__ == "__main__":
    train_rlhf_policy()