#!/usr/bin/env python3
# filepath: /Users/mary/Desktop/Desktop - Mary's Macbook/thesis/codebase/online_rlhf/run_comparative_study.py
"""
Run multiple experiments comparing uncertainty-based and random query selection.
This script runs the OnlineRLHF algorithm 10 times:
- 5 runs with uncertainty-based query selection (use_uncertainty=True)
- 5 runs with random query selection (use_uncertainty=False)
"""
import os
import sys
import copy
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import from the RLHF module
from mcdropout_test import OnlineRLHF, CONFIG
from save_experiments import save_evaluation_results

def run_experiment(use_uncertainty, run_id, max_time_hours=5):
    """Run a single experiment with the given uncertainty setting"""
    start_time = time.time()
    max_time_seconds = max_time_hours * 60 * 60
    
    # Create a copy of the config
    config = copy.deepcopy(CONFIG)
    # Set the uncertainty flag
    config['use_uncertainty'] = use_uncertainty
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uncertainty_str = "uncertainty" if use_uncertainty else "random"
    run_name = f"{uncertainty_str}_run_{run_id}_{timestamp}"
    
    print(f"\n\n{'='*80}\nStarting experiment: {run_name}\n{'='*80}\n")
    
    try:
        # Initialize and run the RLHF algorithm
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rlhf = OnlineRLHF(env_name=config['env_name'], device=device)
        
        # Train with the modified config
        policy, reward_model, results_folder = rlhf.train(
            iterations=config['num_iterations'],
            trajectories_per_iter=config['trajectories_per_iter'],
            trajectories_to_collect=config['trajectories_to_collect'],
            preference_pairs=config['preference_pairs'],
            num_candidate_pairs=config['num_candidate_pairs'],
            reward_epochs=config['reward_epochs'],
            policy_rollouts=config['policy_rollouts'],
            use_uncertainty=config['use_uncertainty'],
            warmup_iterations=config['warmup_iterations'],
            history_pairs_multiplier=config['history_preferences_multiplier']
        )

        # Evaluate the trained policy
        avg_reward, rewards_list = rlhf.evaluate_policy(
            num_episodes=20, 
            max_steps=config['max_steps'], 
            render=False
        )
        
        # Plot evaluation rewards
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rewards_list)+1), rewards_list, 'b-o', linewidth=2)
        plt.title(f'Rewards from Evaluation Episodes ({uncertainty_str})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.tight_layout()
        eval_plot_name = f'evaluation_rewards_{uncertainty_str}_run_{run_id}.png'
        plt.savefig(eval_plot_name)
        
        # Save evaluation results
        save_evaluation_results(results_folder, rewards_list, avg_reward)
        
        # Record the runtime
        runtime = (time.time() - start_time) / 60  # in minutes
        
        # Return results
        return {
            "run_id": run_id,
            "use_uncertainty": use_uncertainty,
            "results_folder": results_folder,
            "avg_reward": avg_reward,
            "runtime_minutes": runtime,
            "timestamp": timestamp,
            "status": "completed",
        }
        
    except Exception as e:
        runtime = (time.time() - start_time) / 60  # in minutes
        print(f"ERROR in experiment {run_name}: {str(e)}")
        return {
            "run_id": run_id,
            "use_uncertainty": use_uncertainty,
            "results_folder": None,
            "avg_reward": None,
            "runtime_minutes": runtime,
            "timestamp": timestamp,
            "status": f"failed: {str(e)}",
        }

def main():
    # Create a directory for the comparative study results
    study_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    study_dir = f"uncertainty_study_{study_timestamp}"
    os.makedirs(study_dir, exist_ok=True)
    
    # Store all results
    all_results = {
        "uncertainty_true": [],
        "uncertainty_false": [],
        "summary": {},
        "study_timestamp": study_timestamp,
    }
    
    # Run 5 experiments with uncertainty=True
    for i in range(10):
        result = run_experiment(use_uncertainty=True, run_id=i+1)
        all_results["uncertainty_true"].append(result)
        
        # Save intermediate results after each run
        with open(os.path.join(study_dir, "results_so_far.json"), "w") as f:
            json.dump(all_results, f, indent=4)
    
    # Run 5 experiments with uncertainty=False
    for i in range(5):
        result = run_experiment(use_uncertainty=False, run_id=i+1)
        all_results["uncertainty_false"].append(result)
        
        # Save intermediate results after each run
        with open(os.path.join(study_dir, "results_so_far.json"), "w") as f:
            json.dump(all_results, f, indent=4)
    
    # Calculate summary statistics
    true_rewards = [r["avg_reward"] for r in all_results["uncertainty_true"] if r["avg_reward"] is not None]
    false_rewards = [r["avg_reward"] for r in all_results["uncertainty_false"] if r["avg_reward"] is not None]
    
    if true_rewards:
        true_avg = np.mean(true_rewards)
        true_std = np.std(true_rewards)
    else:
        true_avg = None
        true_std = None
        
    if false_rewards:
        false_avg = np.mean(false_rewards)
        false_std = np.std(false_rewards)
    else:
        false_avg = None
        false_std = None
    
    # Add summary to results
    all_results["summary"] = {
        "uncertainty_true_avg": true_avg,
        "uncertainty_true_std": true_std,
        "uncertainty_false_avg": false_avg,
        "uncertainty_false_std": false_std,
        "difference": true_avg - false_avg if (true_avg is not None and false_avg is not None) else None
    }
    
    # Save final results
    with open(os.path.join(study_dir, "final_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Print summary
    print("\n\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print("\nResults with uncertainty=True:")
    for i, result in enumerate(all_results["uncertainty_true"]):
        status = result["status"]
        reward = result["avg_reward"]
        folder = result["results_folder"]
        runtime = result["runtime_minutes"]
        if reward:
            print(f"  Run {i+1}: Status={status}, Reward={reward:.2f}, Runtime={runtime:.1f}m, Folder={folder}")
    
    print("\nResults with uncertainty=False:")
    for i, result in enumerate(all_results["uncertainty_false"]):
        status = result["status"]
        reward = result["avg_reward"]
        folder = result["results_folder"]
        runtime = result["runtime_minutes"]
        if reward:
            print(f"  Run {i+1}: Status={status}, Reward={reward:.2f}, Runtime={runtime:.1f}m, Folder={folder}")
    
    # Print averages
    if true_avg is not None:
        print(f"\nUncertainty=True:  Avg Reward = {true_avg:.2f} ± {true_std:.2f}")
    else:
        print("\nUncertainty=True:  No successful runs")
        
    if false_avg is not None:
        print(f"Uncertainty=False: Avg Reward = {false_avg:.2f} ± {false_std:.2f}")
    else:
        print("Uncertainty=False: No successful runs")
    
    if true_avg is not None and false_avg is not None:
        print(f"Difference: {true_avg - false_avg:.2f}")
    
    print(f"\nFull results saved in: {study_dir}/final_results.json")

    # Generate comparative visualization
    if true_rewards and false_rewards:
        plt.figure(figsize=(10, 6))
        plt.bar(['Uncertainty-Based', 'Random'], [true_avg, false_avg], 
                yerr=[true_std, false_std], capsize=10, color=['blue', 'orange'])
        plt.title('Comparison of Query Selection Methods')
        plt.ylabel('Average Reward')
        plt.grid(axis='y')
        plt.savefig(os.path.join(study_dir, 'comparison.png'))

if __name__ == "__main__":
    main()