import matplotlib.pyplot as plt
import numpy as np

def visualize_reward_evolution(iterations, rewards, save_path='reward_evolution.png'):
    """
    Create visualization showing evolution of rewards over training iterations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rewards, 'b-o', linewidth=2)
    plt.title('Reward Evolution During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def visualize_reward_model_loss(iterations, losses, save_path='reward_model_loss.png'):
    """
    Create visualization showing the reward model loss during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, 'r-o', linewidth=2)
    plt.title('Reward Model Loss During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')  # Log scale often helps visualize loss curves
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_reward_correlation(true_rewards, predicted_rewards, save_path='reward_correlation.png'):
    """
    Create visualization showing correlation between true rewards and predicted rewards
    
    Args:
        true_rewards: List of true rewards from the environment
        predicted_rewards: List of predicted rewards from the reward model
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(true_rewards, predicted_rewards, alpha=0.6)
    
    # Find the min and max across both axes to set equal limits
    min_val = min(min(true_rewards), min(predicted_rewards))
    max_val = max(max(true_rewards), max(predicted_rewards))
    
    # Add y=x line for reference (perfect correlation)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (perfect correlation)')
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(true_rewards, predicted_rewards)[0, 1]
    plt.title(f'Correlation Between True and Predicted Rewards (r = {correlation:.3f})')
    plt.xlabel('True Rewards (Environment)')
    plt.ylabel('Predicted Rewards (Reward Model)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_uncertainty_evolution(iterations, random_avgs, active_avgs, save_path='uncertainty_evolution.png'):
    """
    Create visualization showing uncertainty evolution during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, random_avgs, 'r-o', label='Random Sampling')
    plt.plot(iterations, active_avgs, 'b-o', label='Uncertainty Sampling')
    plt.title('Evolution of Uncertainty During Training')
    plt.xlabel('Iteration')
    plt.ylabel('Average Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_uncertainty_comparison(random_uncertainty, active_uncertainty, save_path='uncertainty_comparison.png'):
    """
    Create visualization comparing uncertainty distributions
    """
    plt.figure(figsize=(10, 6))
    plt.hist(random_uncertainty, alpha=0.5, label='Random Sampling')
    plt.hist(active_uncertainty, alpha=0.5, label='Uncertainty Sampling')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution: Random vs Active Sampling')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_preference_accuracy(accuracy_history, save_path='preference_accuracy.png'):
    """
    Create visualization showing preference prediction accuracy
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracy_history)+1), accuracy_history, 'g-o', linewidth=2)
    plt.title('Preference Prediction Accuracy')
    plt.xlabel('Evaluation')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def visualize_evaluation_rewards(rewards_list, save_path='evaluation_rewards.png'):
    """
    Create visualization showing rewards from evaluation episodes
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards_list)+1), rewards_list, 'b-o', linewidth=2)
    plt.title('Rewards from Evaluation Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
