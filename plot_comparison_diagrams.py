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

# Add TensorBoard imports
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False

def get_tensorboard_data(log_dirs, tags=None):
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Please install with: pip install tensorboard")
        return {}
    event_files = []
    for log_dir in log_dirs:
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_files.append(os.path.join(root, file))
    if not event_files:
        print(f"No TensorBoard event files found in {log_dirs}")
        return {}
    all_data = {}
    for event_file in event_files:
        print(f"Processing: {event_file}")
        try:
            if 'EventAccumulator' in globals():
                ea = EventAccumulator(event_file)
                ea.Reload()
                scalar_tags = ea.Tags()['scalars']
                for tag in scalar_tags:
                    if tags is not None and tag not in tags:
                        continue
                    if tag not in all_data:
                        all_data[tag] = []
                    scalar_events = ea.Scalars(tag)
                    for event in scalar_events:
                        all_data[tag].append({
                            'step': event.step,
                            'wall_time': event.wall_time,
                            'value': event.value
                        })
            else:
                # Fallback to summary_iterator
                from tensorflow.python.summary.summary_iterator import summary_iterator
                for event in summary_iterator(event_file):
                    if not event.summary.value:
                        continue
                    for value in event.summary.value:
                        tag = value.tag
                        
                        if tags is not None and tag not in tags:
                            continue
                        
                        if tag not in all_data:
                            all_data[tag] = []
                        
                        if value.HasField('simple_value'):
                            all_data[tag].append({
                                'step': event.step,
                                'wall_time': event.wall_time,
                                'value': value.simple_value
                            })
        
        except Exception as e:
            print(f"Error processing {event_file}: {str(e)}")
            continue
    
    # Convert to DataFrames
    dataframes = {}
    for tag, data_list in all_data.items():
        if data_list:
            dataframes[tag] = pd.DataFrame(data_list)
            dataframes[tag] = dataframes[tag].sort_values('step').reset_index(drop=True)
    
    return dataframes

def get_tensorboard_data_with_file_info(log_dirs, tags=None):
    """
    Extract data from TensorBoard event files with file path information.
    
    Args:
        log_dirs (list): List of paths to directories containing TensorBoard event files
        tags (list): List of specific tags to extract. If None, extract all tags.
    
    Returns:
        dict: Dictionary with tag names as keys and pandas DataFrames as values (including file_path column)
    """
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Please install with: pip install tensorboard")
        return {}
    
    # Find all event files recursively
    event_files = []
    for log_dir in log_dirs:
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dirs}")
        return {}
    
    all_data = {}
    
    for event_file in event_files:
        print(f"Processing: {event_file}")
        
        try:
            # Use EventAccumulator if available (preferred)
            if 'EventAccumulator' in globals():
                ea = EventAccumulator(event_file)
                ea.Reload()
                
                # Get all scalar tags
                scalar_tags = ea.Tags()['scalars']
                
                for tag in scalar_tags:
                    if tags is not None and tag not in tags:
                        continue
                    
                    if tag not in all_data:
                        all_data[tag] = []
                    
                    # Get scalar events
                    scalar_events = ea.Scalars(tag)
                    
                    for event in scalar_events:
                        all_data[tag].append({
                            'step': event.step,
                            'wall_time': event.wall_time,
                            'value': event.value,
                            'file_path': event_file
                        })
            
            else:
                # Fallback to summary_iterator
                from tensorflow.python.summary.summary_iterator import summary_iterator
                
                for event in summary_iterator(event_file):
                    if not event.summary.value:
                        continue
                    
                    for value in event.summary.value:
                        tag = value.tag
                        
                        if tags is not None and tag not in tags:
                            continue
                        
                        if tag not in all_data:
                            all_data[tag] = []
                        
                        if value.HasField('simple_value'):
                            all_data[tag].append({
                                'step': event.step,
                                'wall_time': event.wall_time,
                                'value': value.simple_value,
                                'file_path': event_file
                            })
        
        except Exception as e:
            print(f"Error processing {event_file}: {str(e)}")
            continue
    
    # Convert to DataFrames
    dataframes = {}
    for tag, data_list in all_data.items():
        if data_list:
            dataframes[tag] = pd.DataFrame(data_list)
            dataframes[tag] = dataframes[tag].sort_values('step').reset_index(drop=True)
    
    return dataframes

def analyze_rollout_ep_true_mean(log_dirs, legend_names=None, save_plot=True):
    """
    Analyze the rollout/ep_true_mean metric for different directories.
    
    Args:
        log_dirs (list): List of paths to directories containing TensorBoard event files
        legend_names (list): Optional list of names for the legend.
        save_plot (bool): Whether to save the plot to file
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
    
    # Extract data with file path information
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
    colors = plt.cm.get_cmap('tab10', len(log_dirs))

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
            plt.plot(stats_dir['step'], stats_dir['mean'], 
                     label=legend_name, color=colors(i), linewidth=2)
            plt.fill_between(stats_dir['step'], 
                             stats_dir['ci_lower'], 
                             stats_dir['ci_upper'], 
                             alpha=0.3, color=colors(i))

    plt.xlabel('Step')
    plt.ylabel('Episode Reward Mean')
    plt.title('Evolution of Episode Reward Mean')
    plt.legend()
    plt.xlim(right=2e6)
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

def list_tensorboard_tags(log_dirs):
    """List all available tags in TensorBoard event files"""
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Please install with: pip install tensorboard")
        return set()
    
    event_files = []
    for log_dir in log_dirs:
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_files.append(os.path.join(root, file))
    
    tags = set()
    
    for event_file in event_files:
        try:
            if 'EventAccumulator' in globals():
                ea = EventAccumulator(event_file)
                ea.Reload()
                tags.update(ea.Tags()['scalars'])
            else:
                from tensorflow.python.summary.summary_iterator import summary_iterator
                for event in summary_iterator(event_file):
                    if event.summary.value:
                        for value in event.summary.value:
                            tags.add(value.tag)
        except Exception as e:
            print(f"Error reading {event_file}: {str(e)}")
            continue
    
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
    
    args = parser.parse_args()
    
    if args.analyze_rollout:
        analyze_rollout_ep_true_mean(args.log_dirs, legend_names=args.legend_names)
        return
    
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