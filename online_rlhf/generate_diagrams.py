#!/usr/bin/env python3
"""
This script reads evaluation rewards from multiple experiment folders across multiple result files,
computes averages for uncertainty=True and uncertainty=False runs,
and creates a comparison plot with confidence intervals.

It uses results JSON files to identify experiment folders and conditions.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
import argparse

def read_eval_rewards(folder_path):
    """Read eval_rewards.csv from the metrics subfolder"""
    csv_path = os.path.join(folder_path, 'metrics', 'eval_rewards.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Return the rewards column as a numpy array
            return df['eval_rewards'].values
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return None
    else:
        print(f"File not found: {csv_path}")
        return None

# Calculate confidence interval
def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for data"""
    # For each epoch, calculate confidence interval across runs
    n = len(data)
    if n <= 1:
        return np.zeros(data[0].shape)
    
    # Stack all runs by epoch
    stacked = np.vstack(data)
    # Calculate mean and std for each epoch
    means = np.mean(stacked, axis=0)
    se = stats.sem(stacked, axis=0)
    # Calculate confidence interval
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return h

def process_results_file(results_file_path):
    """Process a single results file and return experiment data"""
    print(f"Reading results from: {results_file_path}")
    
    try:
        with open(results_file_path, 'r') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Error reading {results_file_path}: {e}")
        return None, None, None
    
    # Extract experiment folders for both conditions
    uncertainty_true_folders = [run['results_folder'] for run in results_data['uncertainty_true'] 
                               if run['status'] == 'completed']
    uncertainty_false_folders = [run['results_folder'] for run in results_data['uncertainty_false'] 
                                if run['status'] == 'completed']
    
    print(f"Found {len(uncertainty_true_folders)} experiments with uncertainty=True")
    print(f"Found {len(uncertainty_false_folders)} experiments with uncertainty=False")
    
    return uncertainty_true_folders, uncertainty_false_folders, results_data

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generate reward evolution diagrams from multiple experiment sets')
    parser.add_argument('--results_files', nargs='+', required=True, 
                        help='List of result JSON files to process')
    parser.add_argument('--output', default='reward_evolution_comparison.png',
                        help='Output file name for the plot (default: reward_evolution_comparison.png)')
    args = parser.parse_args()
    
    # Path to the results JSON files
    results_file_paths = args.results_files
    print(f"Processing results files: {results_file_paths}")
    
    # Collect all folders from all results files
    all_uncertainty_true_folders = []
    all_uncertainty_false_folders = []
    all_results_data = []
    
    for file_path in results_file_paths:
        true_folders, false_folders, results_data = process_results_file(file_path)
        if true_folders and false_folders:
            all_uncertainty_true_folders.extend(true_folders)
            all_uncertainty_false_folders.extend(false_folders)
            all_results_data.append(results_data)
    
    print(f"Total experiments with uncertainty=True across all files: {len(all_uncertainty_true_folders)}")
    print(f"Total experiments with uncertainty=False across all files: {len(all_uncertainty_false_folders)}")
    
    # Read eval_rewards for each folder
    uncertainty_true_rewards = []
    uncertainty_false_rewards = []
    
    for folder in all_uncertainty_true_folders:
        rewards = read_eval_rewards(folder)
        if rewards is not None:
            uncertainty_true_rewards.append(rewards)
    
    for folder in all_uncertainty_false_folders:
        rewards = read_eval_rewards(folder)
        if rewards is not None:
            uncertainty_false_rewards.append(rewards)
    
    print(f"Successfully read rewards from {len(uncertainty_true_rewards)} uncertainty=True experiments")
    print(f"Successfully read rewards from {len(uncertainty_false_rewards)} uncertainty=False experiments")
    
    # Ensure all reward arrays have the same length (use min length)
    if uncertainty_true_rewards:
        min_length_true = min(len(rewards) for rewards in uncertainty_true_rewards)
        uncertainty_true_rewards = [rewards[:min_length_true] for rewards in uncertainty_true_rewards]
    
    if uncertainty_false_rewards:
        min_length_false = min(len(rewards) for rewards in uncertainty_false_rewards)
        uncertainty_false_rewards = [rewards[:min_length_false] for rewards in uncertainty_false_rewards]
    
    # Calculate mean and confidence intervals
    if uncertainty_true_rewards:
        uncertainty_true_mean = np.mean(uncertainty_true_rewards, axis=0)
        uncertainty_true_ci = confidence_interval(uncertainty_true_rewards)
    else:
        print("No uncertainty=True data found")
        uncertainty_true_mean = np.array([])
        uncertainty_true_ci = np.array([])
    
    if uncertainty_false_rewards:
        uncertainty_false_mean = np.mean(uncertainty_false_rewards, axis=0)
        uncertainty_false_ci = confidence_interval(uncertainty_false_rewards)
    else:
        print("No uncertainty=False data found")
        uncertainty_false_mean = np.array([])
        uncertainty_false_ci = np.array([])
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    if len(uncertainty_true_mean) > 0:
        epochs_true = np.arange(1, len(uncertainty_true_mean) + 1)
        plt.plot(epochs_true, uncertainty_true_mean, 'b-', linewidth=2, label='With Uncertainty Sampling')
        plt.fill_between(epochs_true, 
                          uncertainty_true_mean - uncertainty_true_ci,
                          uncertainty_true_mean + uncertainty_true_ci,
                          color='blue', alpha=0.2)
    
    if len(uncertainty_false_mean) > 0:
        epochs_false = np.arange(1, len(uncertainty_false_mean) + 1)
        plt.plot(epochs_false, uncertainty_false_mean, 'orange', linewidth=2, label='Random Sampling')
        plt.fill_between(epochs_false,
                          uncertainty_false_mean - uncertainty_false_ci,
                          uncertainty_false_mean + uncertainty_false_ci,
                          color='orange', alpha=0.2)
    
    plt.title('Average Reward Evolution: Uncertainty vs. Random Sampling', fontsize=16)
    plt.xlabel('Training Iteration', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Use integer ticks for x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Calculate aggregate statistics
    true_rewards_final = []
    false_rewards_final = []
    
    for results_data in all_results_data:
        true_rewards = [run['avg_reward'] for run in results_data['uncertainty_true'] 
                       if run['status'] == 'completed' and run['avg_reward'] is not None]
        false_rewards = [run['avg_reward'] for run in results_data['uncertainty_false'] 
                        if run['status'] == 'completed' and run['avg_reward'] is not None]
        
        true_rewards_final.extend(true_rewards)
        false_rewards_final.extend(false_rewards)
    
    # Calculate aggregate summary statistics
    if true_rewards_final:
        true_avg = np.mean(true_rewards_final)
        true_std = np.std(true_rewards_final)
    else:
        true_avg = None
        true_std = None
        
    if false_rewards_final:
        false_avg = np.mean(false_rewards_final)
        false_std = np.std(false_rewards_final)
    else:
        false_avg = None
        false_std = None
    
    # Calculate difference if both averages exist
    difference = true_avg - false_avg if (true_avg is not None and false_avg is not None) else None
    
    # Add text with number of runs and final statistics
    text_str = (f"Uncertainty runs: {len(uncertainty_true_rewards)}\n"
                f"Random runs: {len(uncertainty_false_rewards)}\n\n")
    
    if true_avg is not None:
        text_str += f"Final avg (Uncertainty): {true_avg:.2f} ± {true_std:.2f}\n"
    if false_avg is not None:
        text_str += f"Final avg (Random): {false_avg:.2f} ± {false_std:.2f}\n"
    if difference is not None:
        text_str += f"Difference: {difference:.2f}"
    
    plt.text(0.02, 0.98, text_str,
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    output_path = args.output
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Save numerical data to CSV for reference
    if len(uncertainty_true_mean) > 0 and len(uncertainty_false_mean) > 0:
        # Use the shorter of the two arrays for the CSV
        min_length = min(len(uncertainty_true_mean), len(uncertainty_false_mean))
        
        data = {
            'iteration': np.arange(1, min_length + 1),
            'uncertainty_true_mean': uncertainty_true_mean[:min_length],
            'uncertainty_true_ci': uncertainty_true_ci[:min_length],
            'uncertainty_false_mean': uncertainty_false_mean[:min_length],
            'uncertainty_false_ci': uncertainty_false_ci[:min_length]
        }
        
        csv_output = os.path.splitext(output_path)[0] + '.csv'
        df = pd.DataFrame(data)
        df.to_csv(csv_output, index=False)
        print(f"Numerical data saved to {csv_output}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()