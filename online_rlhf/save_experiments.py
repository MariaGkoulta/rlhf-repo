import os
import json
import csv
import datetime
import shutil
import numpy as np

def save_experiment_results(config, metrics_data, diagrams_path=None):
    """
    Save experiment results, config and metrics to a folder
    
    Args:
        config: The configuration dictionary
        metrics_data: Dictionary containing all metrics
        diagrams_path: Path to diagrams (if already saved elsewhere)
    """
    # Create a timestamp-based folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"results/experiment_results_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # Create a serializable copy of the config dictionary
    serializable_config = {}
    for key, value in config.items():
        try:
            # Test if the value is JSON serializable
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, OverflowError):
            # If not serializable, convert to string representation
            serializable_config[key] = str(value)
    
    # Save config to JSON file
    with open(os.path.join(folder_name, "config.json"), "w") as f:
        json.dump(serializable_config, f, indent=4)
    
    # Create metrics directory
    metrics_dir = os.path.join(folder_name, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save each metric to a separate CSV file
    for metric_name, values in metrics_data.items():
        if isinstance(values, list) and len(values) > 0:
            metric_file = os.path.join(metrics_dir, f"{metric_name}.csv")
            
            with open(metric_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["index", metric_name])
                
                # Write data rows
                for i, value in enumerate(values):
                    writer.writerow([i, value])
                    
            print(f"Saved metric {metric_name} to {metric_file}")
    
    # Copy diagrams to the results folder
    diagram_files = [
        'reward_evolution.png',
        'reward_model_loss.png',
        'reward_correlation.png',
        'preference_accuracy.png',
        'reward_mse_losses.png',
        'correlation_coefficients.png',
        'model_uncertainties.png',
        'evaluation_rewards.png',
        'random_uncertainty_variances.png',
        'uncertainty_heatmap.png',
        'information_gain.png',
    ]
    
    for diagram in diagram_files:
        if os.path.exists(diagram):
            # move instead of copy
            shutil.move(diagram, os.path.join(folder_name, diagram))
    
    print(f"Experiment results saved to {folder_name}/")
    return folder_name


def save_evaluation_results(results_folder, rewards_list, avg_reward):
    """
    Save policy evaluation results in the experiment results folder
    
    Args:
        results_folder: Path to the experiment results folder
        rewards_list: List of rewards from evaluation episodes
        avg_reward: Average reward achieved during evaluation
    """
    # Save evaluation metrics
    eval_data = {
        "episode_rewards": rewards_list,
        "average_reward": avg_reward,
        "num_episodes": len(rewards_list),
        "max_reward": max(rewards_list),
        "min_reward": min(rewards_list),
        "reward_std": float(np.std(rewards_list)),
        "evaluation_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON file
    with open(os.path.join(results_folder, "evaluation_results.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    
    # Copy evaluation_rewards.png to the folder (may be redundant if already copied by save_experiment_results)
    if os.path.exists("evaluation_rewards.png"):
        shutil.move("evaluation_rewards.png", os.path.join(results_folder, "evaluation_rewards.png"))
    
    print(f"Evaluation results saved to {results_folder}/evaluation_results.json")
    return results_folder