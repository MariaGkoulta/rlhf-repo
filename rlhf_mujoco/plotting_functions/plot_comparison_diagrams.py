import os
import glob
import pandas as pd
import numpy as np
import json
import pickle
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tb_utils import (
    load_tensorboard_scalars,
    build_time_series_by_queries,
    plot_time_series_by_queries,
)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

def get_tensorboard_data(log_dirs, tags=None):
    """Delegate TensorBoard scalar extraction to tb_utils for each directory and merge.

    Returns dict[tag] -> DataFrame(step, value, wall_time)
    """
    combined = {}
    for log_dir in log_dirs:
        dfs = load_tensorboard_scalars(log_dir, tags=tags)
        for tag, df in dfs.items():
            # Keep a minimal schema to match the original behavior
            cols = [c for c in ['step', 'wall_time', 'value'] if c in df.columns]
            part = df[cols].copy()
            if tag not in combined:
                combined[tag] = [part]
            else:
                combined[tag].append(part)
    return {tag: pd.concat(parts, ignore_index=True).sort_values('step').reset_index(drop=True)
            for tag, parts in combined.items()}

def get_tensorboard_data_with_file_info(log_dirs, tags=None):
    """Use tb_utils to load per-directory scalars and merge, keeping file_path.

    Returns dict[tag] -> DataFrame(step, value, wall_time, file_path)
    """
    combined = {}
    for log_dir in log_dirs:
        dfs = load_tensorboard_scalars(log_dir, tags=tags)
        for tag, df in dfs.items():
            cols = [c for c in ['step', 'wall_time', 'value', 'file_path'] if c in df.columns]
            part = df[cols].copy()
            if tag not in combined:
                combined[tag] = [part]
            else:
                combined[tag].append(part)
    return {tag: pd.concat(parts, ignore_index=True).sort_values('step').reset_index(drop=True)
            for tag, parts in combined.items()}

def analyze_rollout_ep_true_mean(log_dirs, legend_names=None, save_plot=True, colors=None):
    """
    Analyze the rollout/ep_true_mean metric for different directories.
    
    Args:
        log_dirs (list): List of paths to directories containing TensorBoard event files
        legend_names (list): Optional list of names for the legend.
        save_plot (bool): Whether to save the plot to file
        colors (list): Optional list of colors for the plot lines.
    """
    # Determine which metric to use based on whether event files contain "ground_truth"
    metrics_to_extract = set()
    dir_metric_map = {}
    
    for log_dir in log_dirs:
        # Check event files within this directory to determine the metric
        has_ground_truth = False
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_file_path = os.path.join(root, file)
                    if "ground_truth" in event_file_path.lower():
                        has_ground_truth = True
                        break
            if has_ground_truth:
                break
        
        if has_ground_truth:
            metric = 'rollout/ep_rew_mean'
        else:
            metric = 'rollout/ep_true_mean'
        
        metrics_to_extract.add(metric)
        dir_metric_map[log_dir] = metric
        print(f"Directory: {log_dir} -> Metric: {metric} (ground_truth found: {has_ground_truth})")
    
    # Extract data with file path information (delegates to tb_utils)
    data = get_tensorboard_data_with_file_info(log_dirs, tags=list(metrics_to_extract))
    
    # Check if we have any of the required metrics
    available_metrics = [metric for metric in metrics_to_extract if metric in data]
    if not available_metrics:
        print(f"No required metrics found in the data. Looking for: {list(metrics_to_extract)}")
        return
    
    all_stats = {}

    # Calculate statistics for each step
    def calculate_step_statistics(df_category):
        if df_category.empty:
            return pd.DataFrame()
        stats = df_category.groupby('step')['value'].agg(['mean', 'std', 'count']).reset_index()
        stats['std'] = stats['std'].fillna(0) # Treat std of a single point as 0
        stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
        stats['ci_lower'] = stats['mean'] - 1.96 * stats['stderr']
        stats['ci_upper'] = stats['mean'] + 1.96 * stats['stderr']
        return stats

    plt.figure(figsize=(12, 8))
    
    plot_colors = colors
    if plot_colors is None:
        plot_colors = plt.cm.get_cmap('tab10', len(log_dirs))

    for i, log_dir in enumerate(log_dirs):
        legend_name = legend_names[i] if legend_names else os.path.basename(os.path.normpath(log_dir))
        # Normalize path for consistent matching
        normalized_log_dir = os.path.normpath(log_dir)
        
        # Get the appropriate metric for this directory
        metric = dir_metric_map[log_dir]
        
        if metric not in data:
            print(f"No data found for metric {metric} in directory: {log_dir}")
            continue
        
        df = data[metric]
        
        # Filter dataframe for files within the current log_dir
        dir_mask = df['file_path'].apply(lambda x: os.path.normpath(x).startswith(normalized_log_dir))
        df_dir = df[dir_mask].copy()
        
        if df_dir.empty:
            print(f"No data found for directory: {log_dir}")
            continue

        print(f"Processing directory: {log_dir} ({df_dir['file_path'].nunique()} unique files) using metric: {metric}")
        
        stats_dir = calculate_step_statistics(df_dir)
        stats_dir['category'] = legend_name
        all_stats[legend_name] = stats_dir
        
        if not stats_dir.empty:
            color = plot_colors[i] if isinstance(plot_colors, list) else plot_colors(i)
            plt.plot(stats_dir['step'], stats_dir['mean'], 
                     label=legend_name, color=color, linewidth=2)
            plt.fill_between(stats_dir['step'], 
                             stats_dir['ci_lower'], 
                             stats_dir['ci_upper'], 
                             alpha=0.3, color=color)

    plt.xlabel('Step')
    plt.ylabel('Episode True Reward Mean')
    plt.title('Evolution of Episode True Reward Mean')
    plt.legend()
    plt.xlim(right=1e6)
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig('rollout_ep_true_mean_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'rollout_ep_true_mean_comparison.png'")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for name, stats_df in all_stats.items():
        if not stats_df.empty:
            # Need to find the number of runs from the original filtered dataframe
            corresponding_log_dir = log_dirs[legend_names.index(name)] if legend_names and name in legend_names else next((d for d in log_dirs if os.path.basename(os.path.normpath(d)) == name), log_dirs[0])
            metric = dir_metric_map[corresponding_log_dir]
            num_runs = data[metric][data[metric]['file_path'].apply(lambda x: os.path.normpath(x).startswith(os.path.normpath(corresponding_log_dir)))].file_path.nunique()
            print(f"{name}:")
            print(f"  Final mean: {stats_df['mean'].iloc[-1]:.4f}")
            print(f"  Final std: {stats_df['std'].iloc[-1]:.4f}")
            print(f"  Number of runs: {num_runs}")
        else:
            print(f"{name}: No data found.")

    return all_stats


def analyze_rollout_true_mean_vs_queries(log_dirs, legend_names=None, save_plot=True, colors=None):
    """Same as analyze_rollout_ep_true_mean but with x-axis as number of queries.

    For each directory (series):
    - Determine reward metric (ep_rew_mean if ground_truth present else ep_true_mean).
    - For each run, align reward values at a step to the latest queries count at or before that step.
    - Aggregate across runs by step to compute mean reward and 95% CI, and compute the mean queries at that step.
    - Plot x = mean queries per step, y = mean reward per step, with CI shaded (y-dimension).
    """
    # Prepare color map similar to analyze_rollout_ep_true_mean
    color_map = None
    if colors is not None and legend_names is not None:
        if isinstance(colors, list):
            color_map = {legend_names[i]: c for i, c in enumerate(colors) if i < len(legend_names)}
        elif isinstance(colors, dict):
            color_map = colors

    # Collect per-series aggregated points
    series_results = {}
    plt.figure(figsize=(12, 8))

    plot_colors = colors
    if plot_colors is None:
        plot_colors = plt.cm.get_cmap('tab10', len(log_dirs))

    for i, log_dir in enumerate(log_dirs):
        # Determine metric based on ground_truth presence in file paths
        has_ground_truth = False
        for root, _, files in os.walk(log_dir):
            if any(
                f.startswith("events.out.tfevents") and "ground_truth" in os.path.join(root, f).lower()
                for f in files
            ):
                has_ground_truth = True
                break
        metric = 'rollout/ep_rew_mean' if has_ground_truth else 'rollout/ep_true_mean'
        legend_name = legend_names[i] if legend_names else os.path.basename(os.path.normpath(log_dir))

        # Load data for this directory
        dfs = load_tensorboard_scalars(log_dir, tags=[metric, 'params/num_train_data'])
        if metric not in dfs or 'params/num_train_data' not in dfs:
            print(f"Skipping {log_dir}: required tags missing ({metric} and/or params/num_train_data)")
            continue

        mdf = dfs[metric][['run_id', 'step', 'value']].copy().sort_values(['run_id', 'step'])
        qdf = dfs['params/num_train_data'][['run_id', 'step', 'value']].rename(columns={'value': 'queries'})
        qdf = qdf.sort_values(['run_id', 'step'])

        print(mdf)
        print(qdf)

        # Per-run alignment: map each reward at a step to the latest queries value at or before that step
        aligned_points = []  # rows: {run_id, queries, value}
        for rid in sorted(set(mdf['run_id']).intersection(set(qdf['run_id']))):
            m_run = mdf[mdf['run_id'] == rid][['step', 'value']].sort_values('step').reset_index(drop=True)
            q_run = qdf[qdf['run_id'] == rid][['step', 'queries']].sort_values('step').reset_index(drop=True)
            if m_run.empty or q_run.empty:
                continue
            merged = pd.merge_asof(
                m_run,
                q_run,
                on='step',
                direction='backward'
            ).dropna(subset=['queries'])
            if merged.empty:
                continue
            # Collapse any duplicate rewards mapping to the same queries value within a run
            # Use integer queries when possible to avoid float grouping artifacts
            merged['queries'] = merged['queries'].round().astype(int)
            collapsed = merged.groupby('queries', as_index=False)['value'].mean()
            collapsed['run_id'] = rid
            aligned_points.append(collapsed[['run_id', 'queries', 'value']])

        if not aligned_points:
            print(f"No aligned data for directory: {log_dir}")
            continue

        aligned = pd.concat(aligned_points, ignore_index=True)

        # Aggregate across runs by queries: mean reward and 95% CI per queries count
        stats = aligned.groupby('queries').agg(
            mean=('value', 'mean'),
            std=('value', 'std'),
            count=('value', 'count')
        ).reset_index()
        stats['std'] = stats['std'].fillna(0.0)
        stats['stderr'] = stats.apply(lambda r: (r['std'] / np.sqrt(r['count'])) if r['count'] > 0 else 0.0, axis=1)
        stats['ci_lower'] = stats['mean'] - 1.96 * stats['stderr']
        stats['ci_upper'] = stats['mean'] + 1.96 * stats['stderr']

        # Sort by queries for plotting on x-axis
        stats = stats.sort_values('queries').reset_index(drop=True)
        series_results[legend_name] = {
            'stats': stats,
            'metric': metric,
            'num_runs': aligned['run_id'].nunique(),
        }

        # Plot
        color = None
        if isinstance(plot_colors, list) and i < len(plot_colors):
            color = plot_colors[i]
        elif not isinstance(plot_colors, list):
            color = plot_colors(i)
        if color_map and legend_name in color_map:
            color = color_map[legend_name]

    plt.plot(stats['queries'], stats['mean'], label=legend_name, color=color, linewidth=2)
    plt.fill_between(stats['queries'], stats['ci_lower'], stats['ci_upper'], alpha=0.3, color=color)

    # Finalize plot
    plt.xlabel('Number of Queries')
    plt.ylabel('Episode True Reward Mean')
    plt.title('Episode True Reward Mean vs Number of Queries')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_plot:
        plt.savefig('rollout_ep_true_mean_vs_queries.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'rollout_ep_true_mean_vs_queries.png'")

    plt.show()

    # Print summary statistics similar to analyze_rollout_ep_true_mean
    print("\nSummary Statistics (reward vs queries):")
    for name, info in series_results.items():
        stats = info['stats']
        if not stats.empty:
            print(f"{name}:")
            print(f"  Final mean reward: {stats['mean'].iloc[-1]:.4f}")
            print(f"  Final std: {stats['std'].iloc[-1]:.4f}")
            print(f"  Final queries: {int(stats['queries'].iloc[-1])}")
            print(f"  Number of runs: {info['num_runs']}")
        else:
            print(f"{name}: No data found.")

    # Return a combined DataFrame for possible downstream use
    out_frames = []
    for name, info in series_results.items():
        df = info['stats'].copy()
        df['series'] = name
        out_frames.append(df[['series', 'queries', 'mean', 'ci_lower', 'ci_upper', 'count']])
    if out_frames:
        return pd.concat(out_frames, ignore_index=True)
    return None

def list_tensorboard_tags(log_dirs):
    """List all available tags in TensorBoard event files.

    Uses tb_utils.load_tensorboard_scalars per directory and unions the keys.
    """
    tags = set()
    for log_dir in log_dirs:
        dfs = load_tensorboard_scalars(log_dir, tags=None)
        tags.update(dfs.keys())
    if not tags:
        print("No TensorBoard tags found (is tensorboard installed and logs present?)")
    return tags

def get_pytorch_data(log_dirs, file_pattern="*.json", tags=None):
    """
    Extract data from PyTorch training logs.
    
    Args:
        log_dirs (list): List of paths to directories containing log files
        file_pattern (str): Pattern to match log files (*.json, *.csv, *.pkl, etc.)
        tags (list): List of specific metrics to extract. If None, extract all.
    
    Returns:
        dict: Dictionary with metric names as keys and pandas DataFrames as values
    """
    log_files = []
    for log_dir in log_dirs:
        log_files.extend(glob.glob(os.path.join(log_dir, "**", file_pattern), recursive=True))

    if not log_files:
        print(f"No log files found in {log_dirs} matching pattern {file_pattern}")
        return {}
    
    all_data = {}
    
    for log_file in log_files:
        print(f"Processing: {log_file}")
        
        try:
            # Handle different file formats
            if log_file.endswith('.json'):
                data = read_json_logs(log_file)
            elif log_file.endswith('.csv'):
                data = read_csv_logs(log_file)
            elif log_file.endswith('.pkl') or log_file.endswith('.pickle'):
                data = read_pickle_logs(log_file)
            elif log_file.endswith('.pt') or log_file.endswith('.pth'):
                data = read_torch_logs(log_file)
            else:
                print(f"Unsupported file format: {log_file}")
                continue
            
            # Merge data from this file
            for key, values in data.items():
                if tags is not None and key not in tags:
                    continue
                
                if key not in all_data:
                    all_data[key] = []
                
                all_data[key].extend(values)
                
        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")
            continue
    
    # Convert to DataFrames
    dataframes = {}
    for metric, data_list in all_data.items():
        if data_list:
            dataframes[metric] = pd.DataFrame(data_list)
            if 'epoch' in dataframes[metric].columns:
                dataframes[metric] = dataframes[metric].sort_values('epoch').reset_index(drop=True)
            elif 'step' in dataframes[metric].columns:
                dataframes[metric] = dataframes[metric].sort_values('step').reset_index(drop=True)
    
    return dataframes

def read_json_logs(log_file):
    """Read JSON log files (common format for PyTorch training logs)"""
    data = {}
    
    with open(log_file, 'r') as f:
        # Handle both single JSON object and line-separated JSON
        content = f.read().strip()
        
        if content.startswith('['):
            # Single JSON array
            logs = json.loads(content)
        else:
            # Line-separated JSON
            logs = []
            for line in content.split('\n'):
                if line.strip():
                    logs.append(json.loads(line))
    
    # Extract metrics
    for log_entry in logs:
        for key, value in log_entry.items():
            if isinstance(value, (int, float)):
                if key not in data:
                    data[key] = []
                
                entry = {'value': value}
                if 'epoch' in log_entry:
                    entry['epoch'] = log_entry['epoch']
                if 'step' in log_entry:
                    entry['step'] = log_entry['step']
                
                data[key].append(entry)
    
    return data

def read_csv_logs(log_file):
    """Read CSV log files"""
    df = pd.read_csv(log_file)
    data = {}
    
    for column in df.columns:
        if column in ['epoch', 'step']:
            continue
        
        data[column] = []
        for idx, row in df.iterrows():
            entry = {'value': row[column]}
            if 'epoch' in df.columns:
                entry['epoch'] = row['epoch']
            if 'step' in df.columns:
                entry['step'] = row['step']
            
            data[column].append(entry)
    
    return data

def read_pickle_logs(log_file):
    """Read pickle log files"""
    with open(log_file, 'rb') as f:
        logs = pickle.load(f)
    
    data = {}
    
    if isinstance(logs, dict):
        # Dictionary format
        for key, values in logs.items():
            if isinstance(values, list):
                data[key] = [{'value': v} for v in values]
            else:
                data[key] = [{'value': values}]
    elif isinstance(logs, list):
        # List of dictionaries
        for log_entry in logs:
            for key, value in log_entry.items():
                if isinstance(value, (int, float)):
                    if key not in data:
                        data[key] = []
                    
                    entry = {'value': value}
                    if 'epoch' in log_entry:
                        entry['epoch'] = log_entry['epoch']
                    if 'step' in log_entry:
                        entry['step'] = log_entry['step']
                    
                    data[key].append(entry)
    
    return data

def read_torch_logs(log_file):
    """Read PyTorch checkpoint files that contain training logs"""
    checkpoint = torch.load(log_file, map_location='cpu')
    data = {}
    
    # Common keys where training logs might be stored
    log_keys = ['train_losses', 'val_losses', 'train_acc', 'val_acc', 'metrics', 'history']
    
    for key in log_keys:
        if key in checkpoint:
            values = checkpoint[key]
            if isinstance(values, list):
                data[key] = [{'value': v} for v in values]
            elif isinstance(values, dict):
                for metric, metric_values in values.items():
                    if isinstance(metric_values, list):
                        data[metric] = [{'value': v} for v in metric_values]
    
    return data

def list_available_metrics(log_dirs, file_pattern="*.json"):
    """List all available metrics in the log files"""
    log_files = []
    for log_dir in log_dirs:
        log_files.extend(glob.glob(os.path.join(log_dir, "**", file_pattern), recursive=True))
    metrics = set()
    
    for log_file in log_files:
        try:
            if log_file.endswith('.json'):
                data = read_json_logs(log_file)
            elif log_file.endswith('.csv'):
                data = read_csv_logs(log_file)
            elif log_file.endswith('.pkl') or log_file.endswith('.pickle'):
                data = read_pickle_logs(log_file)
            elif log_file.endswith('.pt') or log_file.endswith('.pth'):
                data = read_torch_logs(log_file)
            else:
                continue
            
            metrics.update(data.keys())
            
        except Exception as e:
            print(f"Error reading {log_file}: {str(e)}")
            continue
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Extract data from PyTorch training logs')
    parser.add_argument('log_dirs', nargs='+', help='Path(s) to log directories')
    parser.add_argument('--pattern', default='*.json', help='File pattern to match (default: *.json)')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to extract (optional)')
    parser.add_argument('--list-metrics', action='store_true', help='List available metrics and exit')
    parser.add_argument('--tensorboard', action='store_true', help='Read TensorBoard event files')
    parser.add_argument('--analyze-rollout', action='store_true', help='Analyze rollout/ep_true_mean metric')
    parser.add_argument('--legend-names', nargs='+', help='Custom names for legend in rollout analysis')
    parser.add_argument('--colors', nargs='+', help='List of colors for the plot lines')
    parser.add_argument('--plot-reward-vs-queries', action='store_true', help='Plot episode true reward vs number of queries')
    
    args = parser.parse_args()
    
    # Allow running multiple actions in one invocation
    # ran_any_plot = False
    if args.analyze_rollout:
        analyze_rollout_ep_true_mean(args.log_dirs, legend_names=args.legend_names, colors=args.colors)
        # ran_any_plot = True

    # if args.plot_reward_vs_queries:
    #     analyze_rollout_true_mean_vs_queries(args.log_dirs, legend_names=args.legend_names, colors=args.colors)
    #     ran_any_plot = True

    # if ran_any_plot:
    #     return

    if args.tensorboard:
        if args.list_metrics:
            print("Available TensorBoard tags:")
            tags = list_tensorboard_tags(args.log_dirs)
            for tag in sorted(tags):
                print(f"  - {tag}")
            return
        
        # Extract TensorBoard data
        data = get_tensorboard_data(args.log_dirs, args.metrics)
    else:
        if args.list_metrics:
            print("Available metrics:")
            metrics = list_available_metrics(args.log_dirs, args.pattern)
            for metric in sorted(metrics):
                print(f"  - {metric}")
            return
        
        # Extract data
        data = get_pytorch_data(args.log_dirs, args.pattern, args.metrics)
    
    # Display summary
    print(f"\nExtracted data for {len(data)} metrics:")
    for metric, df in data.items():
        print(f"  {metric}: {len(df)} data points")
        if 'epoch' in df.columns:
            print(f"    Epochs: {df['epoch'].min()} to {df['epoch'].max()}")
        if 'step' in df.columns:
            print(f"    Steps: {df['step'].min()} to {df['step'].max()}")
        if 'value' in df.columns:
            print(f"    Values: {df['value'].min():.4f} to {df['value'].max():.4f}")
        print()

if __name__ == "__main__":
    main()