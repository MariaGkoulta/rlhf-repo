import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime
from online_rlhf import OnlineRLHF
import shutil
from visualizations import (
    visualize_reward_evolution,
    visualize_reward_model_loss,
    visualize_reward_correlation,
    visualize_uncertainty_evolution,
    visualize_uncertainty_comparison,
    visualize_preference_accuracy,
    visualize_evaluation_rewards
)

def setup_experiment_directories():
    """Create directory structure for the experiments"""
    # Create main experiments directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiment_results_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directories for each configuration
    configs = [
        "active", 
        "random"
    ]
    
    for config in configs:
        for run in range(1, 6):  # 5 runs per configuration
            run_dir = os.path.join(base_dir, f"{config}/run_{run}")
            os.makedirs(run_dir, exist_ok=True)
    
    # Create a summary directory
    summary_dir = os.path.join(base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    return base_dir

def run_experiment(config_name, use_uncertainty, use_buffer, run_num, base_dir):
    """Run a single experiment with the specified configuration"""
    print(f"\n{'='*80}")
    print(f"Running experiment: {config_name}, Run {run_num}/5")
    print(f"Configuration: uncertainty={use_uncertainty}, buffer={use_buffer}")
    print(f"{'='*80}\n")
    
    # Create directory for this specific run
    run_dir = os.path.join(base_dir, f"{config_name}/run_{run_num}")
    
    # Turn off interactive plotting to avoid showing figures during execution
    plt.ioff()
    
    # Redirect matplotlib to save to the run directory
    original_savefig = plt.savefig
    
    def custom_savefig(fname, *args, **kwargs):
        path = os.path.join(run_dir, fname)
        result = original_savefig(path, *args, **kwargs)
        plt.close()  # Close the figure after saving to avoid memory leaks
        return result
    
    plt.savefig = custom_savefig
    
    # Initialize and run the RLHF algorithm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Smaller parameters for experiments to run faster
    iterations = 80
    trajectories_per_iter = 100  
    
    rlhf = OnlineRLHF(env_name='CartPole-v1', device=device, buffer_capacity=5000)
    policy, reward_model, training_metrics = rlhf.train(
        iterations=iterations,
        trajectories_per_iter=trajectories_per_iter,
        preference_pairs=5,
        reward_epochs=5,
        policy_rollouts=100,
        buffer_sample_ratio=0.8,
        recent_trajectory_ratio=0.2,
        use_uncertainty=use_uncertainty,
        use_buffer=use_buffer,
        return_metrics=True  # Ensure we get metrics back from training
    )
    
    # Evaluate the trained policy
    avg_reward, rewards_list = rlhf.evaluate_policy(num_episodes=10, max_steps=500, render=False)
    print(f"Final evaluation: Average reward = {avg_reward:.2f}")
    
    # Plot the rewards from the evaluation
    visualize_evaluation_rewards(rewards_list)
    
    # Restore original plt.savefig
    plt.savefig = original_savefig
    
    # Close any remaining figures
    plt.close('all')
    
    # Save experiment metadata
    metadata = {
        'configuration': config_name,
        'use_uncertainty': use_uncertainty,
        'use_buffer': use_buffer,
        'run_number': run_num,
        'iterations': iterations,
        'trajectories_per_iter': trajectories_per_iter,
        'avg_final_reward': avg_reward,
        'rewards': rewards_list,
        'epoch_rewards': training_metrics['epoch_rewards'],
        'reward_model_losses': training_metrics['reward_model_losses'],
        'uncertainty_values': training_metrics.get('uncertainty_values', []),
        'preference_accuracies': training_metrics.get('preference_accuracies', [])
    }
    
    # Return experiment results
    return metadata

def run_all_experiments():
    """Run all experiment configurations"""
    # Create directory structure
    base_dir = setup_experiment_directories()
    print(f"Storing experiment results in: {base_dir}")
    
    # Define experiment configurations
    configurations = [
        ('active', True, True),
        ('random', False, True),
    ]
    
    # Store all results
    all_results = []
    
    # Dictionary to store per-config epoch rewards
    config_epoch_rewards = {}
    config_model_losses = {}
    config_uncertainty_values = {}
    config_preference_accuracies = {}
    config_evaluation_rewards = {}
    
    # Run each configuration 5 times
    for config_name, use_uncertainty, use_buffer in configurations:
        config_results = []
        config_epoch_rewards[config_name] = []
        config_model_losses[config_name] = []
        config_uncertainty_values[config_name] = []
        config_preference_accuracies[config_name] = []
        config_evaluation_rewards[config_name] = []
        
        for run in range(1, 6):
            start_time = time.time()
            
            # Run the experiment
            result = run_experiment(
                config_name=config_name,
                use_uncertainty=use_uncertainty,
                use_buffer=use_buffer,
                run_num=run,
                base_dir=base_dir
            )
            
            elapsed_time = time.time() - start_time
            result['elapsed_time'] = elapsed_time
            config_results.append(result)
            all_results.append(result)
            
            # Store metrics for averaging later
            config_epoch_rewards[config_name].append(result['epoch_rewards'])
            config_model_losses[config_name].append(result['reward_model_losses'])
            config_uncertainty_values[config_name].append(result['uncertainty_values'])
            config_preference_accuracies[config_name].append(result['preference_accuracies'])
            config_evaluation_rewards[config_name].append(result['rewards'])
            
            print(f"Completed {config_name} run {run}/5 in {elapsed_time:.2f} seconds")
        
        # Calculate aggregate metrics for this configuration
        avg_reward = np.mean([r['avg_final_reward'] for r in config_results])
        std_reward = np.std([r['avg_final_reward'] for r in config_results])
        print(f"\n{config_name} - Average reward across 5 runs: {avg_reward:.2f} ± {std_reward:.2f}")
    
    # Create a summary report
    create_summary_report(all_results, base_dir)
    
    # Generate summary visualizations
    create_summary_visualizations(
        base_dir,
        config_epoch_rewards,
        config_model_losses,
        config_uncertainty_values,
        config_preference_accuracies,
        config_evaluation_rewards
    )

def create_summary_visualizations(base_dir, epoch_rewards, model_losses, uncertainty_values, 
                                 preference_accuracies, evaluation_rewards):
    """Create summary visualizations for all experiment configurations"""
    summary_dir = os.path.join(base_dir, "summary")
    
    # Turn off interactive plotting
    plt.ioff()
    
    # 1. Average reward evolution across epochs for each config
    plt.figure(figsize=(12, 6))
    for config, rewards_list in epoch_rewards.items():
        # Convert list of lists to numpy array for easier manipulation
        rewards_array = np.array(rewards_list)
        # Calculate mean and std across runs for each epoch
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        
        # Plot with error bands
        epochs = np.arange(len(mean_rewards))
        plt.plot(epochs, mean_rewards, label=config)
        plt.fill_between(epochs, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Reward Evolution Across Configurations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'avg_reward_evolution.png'))
    plt.close()
    
    # 2. Average reward model loss across epochs for each config
    plt.figure(figsize=(12, 6))
    for config, losses_list in model_losses.items():
        losses_array = np.array(losses_list)
        mean_losses = np.mean(losses_array, axis=0)
        std_losses = np.std(losses_array, axis=0)
        
        epochs = np.arange(len(mean_losses))
        plt.plot(epochs, mean_losses, label=config)
        plt.fill_between(epochs, mean_losses - std_losses, mean_losses + std_losses, alpha=0.3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reward Model Loss Across Configurations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'avg_reward_model_loss.png'))
    plt.close()
    
    # 3. Average uncertainty evolution for each config
    plt.figure(figsize=(12, 6))
    for config, uncertainties_list in uncertainty_values.items():
        if not uncertainties_list[0]:  # Skip if empty (no uncertainty configurations)
            continue
            
        uncertainties_array = np.array(uncertainties_list)
        mean_uncertainties = np.mean(uncertainties_array, axis=0)
        std_uncertainties = np.std(uncertainties_array, axis=0)
        
        epochs = np.arange(len(mean_uncertainties))
        plt.plot(epochs, mean_uncertainties, label=config)
        plt.fill_between(epochs, mean_uncertainties - std_uncertainties, 
                         mean_uncertainties + std_uncertainties, alpha=0.3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty Evolution Across Configurations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'avg_uncertainty_evolution.png'))
    plt.close()
    
    # 4. Average preference accuracy for each config
    plt.figure(figsize=(12, 6))
    for config, accuracies_list in preference_accuracies.items():
        if not accuracies_list[0]:  # Skip if empty
            continue
            
        accuracies_array = np.array(accuracies_list)
        mean_accuracies = np.mean(accuracies_array, axis=0)
        std_accuracies = np.std(accuracies_array, axis=0)
        
        epochs = np.arange(len(mean_accuracies))
        plt.plot(epochs, mean_accuracies, label=config)
        plt.fill_between(epochs, mean_accuracies - std_accuracies, 
                         mean_accuracies + std_accuracies, alpha=0.3)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Preference Accuracy Across Configurations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'avg_preference_accuracy.png'))
    plt.close()
    
    # 5. Reward model comparison (bar chart)
    plt.figure(figsize=(10, 6))
    configs = list(epoch_rewards.keys())
    # Fix: Convert to numpy array and then extract the last element for each run
    final_rewards = [np.mean(np.array(epoch_rewards[c])[:, -1]) for c in configs]
    std_rewards = [np.std(np.array(epoch_rewards[c])[:, -1]) for c in configs]
    
    plt.bar(configs, final_rewards, yerr=std_rewards, capsize=10)
    plt.ylabel('Final Reward')
    plt.title('Final Reward Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'final_reward_comparison.png'))
    plt.close()
    
    # 6. Policy evaluation comparison across configs
    plt.figure(figsize=(12, 6))
    box_data = []
    for config in configs:
        # Flatten all evaluation rewards for this config
        all_rewards = [item for sublist in evaluation_rewards[config] for item in sublist]
        box_data.append(all_rewards)
    
    plt.boxplot(box_data, labels=configs)
    plt.ylabel('Evaluation Reward')
    plt.title('Policy Evaluation Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'policy_evaluation_comparison.png'))
    plt.close()
    
    # 7. Average learning curves (smoothed)
    plt.figure(figsize=(12, 6))
    window_size = 5  # for smoothing
    
    for config, rewards_list in epoch_rewards.items():
        rewards_array = np.array(rewards_list)
        mean_rewards = np.mean(rewards_array, axis=0)
        
        # Simple moving average for smoothing
        smoothed = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
        epochs = np.arange(len(smoothed))
        plt.plot(epochs, smoothed, label=config)
    
    plt.xlabel('Epoch')
    plt.ylabel('Smoothed Average Reward')
    plt.title('Smoothed Learning Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'smoothed_learning_curves.png'))
    plt.close()
    
    # 8. Uncertainty vs Reward correlation
    plt.figure(figsize=(12, 6))
    for config in configs:
        if 'uncertainty' in config:  # Only for configurations with uncertainty
            mean_rewards = np.mean(np.array(epoch_rewards[config]), axis=0)
            mean_uncertainty = np.mean(np.array(uncertainty_values[config]), axis=0)
            
            if len(mean_uncertainty) > 0:
                plt.scatter(mean_uncertainty, mean_rewards, label=config, alpha=0.7)
    
    if any('uncertainty' in c for c in configs):
        plt.xlabel('Uncertainty')
        plt.ylabel('Reward')
        plt.title('Uncertainty vs Reward Correlation')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'uncertainty_reward_correlation.png'))
    plt.close()
    
    print(f"Summary visualizations saved to {summary_dir}")

def create_summary_report(results, base_dir):
    """Create a summary report of all experiments"""
    df = pd.DataFrame(results)
    
    # Create summary by configuration
    summary = df.groupby(['configuration']).agg({
        'avg_final_reward': ['mean', 'std'],
        'elapsed_time': ['mean', 'std']
    })
    
    # Save summary to CSV
    summary_path = os.path.join(base_dir, 'experiment_summary.csv')
    summary.to_csv(summary_path)
    
    # Turn off interactive plotting
    plt.ioff()
    
    # Create a plot comparing the configurations
    plt.figure(figsize=(10, 6))
    
    # Bar plot of rewards by configuration
    configs = df['configuration'].unique()
    rewards = [df[df['configuration'] == c]['avg_final_reward'].mean() for c in configs]
    errors = [df[df['configuration'] == c]['avg_final_reward'].std() for c in configs]
    
    bars = plt.bar(configs, rewards, yerr=errors, capsize=10)
    plt.ylabel('Average Reward')
    plt.title('Comparison of RLHF Configurations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(base_dir, 'configuration_comparison.png'))
    plt.close()
    
    # Save the full results dataframe
    df.to_csv(os.path.join(base_dir, 'all_results.csv'))
    
    print(f"Summary report saved to {base_dir}")

if __name__ == "__main__":
    run_all_experiments()
    print("All experiments completed!")