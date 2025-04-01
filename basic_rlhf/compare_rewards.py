import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from policy_network import PolicyNetwork
from reward_model import RewardModel

def evaluate_rewards(policy_path, episodes=10, device='cpu'):
    # Set up environment and models
    env = gym.make('CartPole-v1')
    policy = PolicyNetwork(state_dim=4).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    
    reward_model = RewardModel(state_dim=4).to(device)
    reward_model.load_state_dict(torch.load('reward_model.pth', map_location=device))
    reward_model.eval()
    
    # Storage for results
    all_true_rewards = []
    all_predicted_rewards = []
    episode_lengths = []
    
    # Run evaluations
    for ep in range(episodes):
        state, _ = env.reset()
        episode_true_rewards = []
        episode_pred_rewards = []
        done = False
        steps = 0
        
        while not done:
            # Get action from policy
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                probs = policy(state_tensor)
                action = torch.multinomial(probs, 1).item()
                
                # Get predicted reward for this state
                pred_reward = reward_model(state_tensor).item()
            
            # Take action in environment
            next_state, true_reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Record rewards
            episode_true_rewards.append(true_reward)
            episode_pred_rewards.append(pred_reward)
            
            # Update state
            state = next_state
            steps += 1
        
        # Store episode results
        all_true_rewards.append(episode_true_rewards)
        all_predicted_rewards.append(episode_pred_rewards)
        episode_lengths.append(steps)
        
        print(f"Episode {ep+1}: Length={steps}, True Return={sum(episode_true_rewards):.2f}, "
              f"Predicted Return={sum(episode_pred_rewards):.4f}")
    
    env.close()
    return all_true_rewards, all_predicted_rewards, episode_lengths

def plot_reward_comparison(true_rewards, pred_rewards, episode_lengths):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Episode Returns
    plt.subplot(2, 2, 1)
    episodes = list(range(1, len(episode_lengths) + 1))
    true_returns = [sum(rewards) for rewards in true_rewards]
    pred_returns = [sum(rewards) for rewards in pred_rewards]
    
    plt.plot(episodes, true_returns, 'b-', label='True Returns')
    plt.plot(episodes, pred_returns, 'r-', label='Predicted Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode Returns: True vs Predicted')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Correlation
    plt.subplot(2, 2, 2)
    scaled_pred_returns = np.array(pred_returns) * (np.mean(true_returns) / np.mean(pred_returns))
    plt.scatter(true_returns, scaled_pred_returns)
    min_val = min(min(true_returns), min(scaled_pred_returns))
    max_val = max(max(true_returns), max(scaled_pred_returns))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('True Returns')
    plt.ylabel('Scaled Predicted Returns')
    plt.title('Correlation: True vs Predicted Returns')
    plt.grid(True)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(true_returns, pred_returns)[0, 1]
    plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Plot 3: Detailed reward sequence for one episode
    plt.subplot(2, 1, 2)
    # Choose the longest episode for detailed view
    longest_ep = episode_lengths.index(max(episode_lengths))
    steps = list(range(1, episode_lengths[longest_ep] + 1))
    
    plt.plot(steps, true_rewards[longest_ep], 'b-', label='True Rewards')
    
    # Scale predicted rewards to have similar magnitude for visualization
    scale_factor = np.mean(true_rewards[longest_ep]) / np.mean(pred_rewards[longest_ep]) if np.mean(pred_rewards[longest_ep]) != 0 else 1
    scaled_pred = [r * scale_factor for r in pred_rewards[longest_ep]]
    plt.plot(steps, scaled_pred, 'r-', label='Scaled Predicted Rewards')
    
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'Detailed Reward Sequence - Episode {longest_ep+1} (Length: {episode_lengths[longest_ep]})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reward_comparison.png')
    plt.show()
    print(f"Plot saved as 'reward_comparison.png'")

if __name__ == "__main__":
    # Test with both policies to compare
    policies = {
        'RLHF Policy': 'rlhf_policy.pth',
        'Baseline Policy': 'baseline_policy.pth'  # Create this by saving your baseline policy
    }
    
    for name, path in policies.items():
        try:
            print(f"\nEvaluating {name}...")
            true_rewards, pred_rewards, episode_lengths = evaluate_rewards(path, episodes=20)
            plot_reward_comparison(true_rewards, pred_rewards, episode_lengths)
        except FileNotFoundError:
            print(f"Policy file {path} not found. Skipping.")