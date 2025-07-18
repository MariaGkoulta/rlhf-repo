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
    """
    Extract data from TensorBoard event files.
    
    Args:
        log_dirs (list): List of paths to directories containing TensorBoard event files
        tags (list): List of specific tags to extract. If None, extract all tags.
    
    Returns:
        dict: Dictionary with tag names as keys and pandas DataFrames as values
    """
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Please install with: pip install tensorboard")
        return {}
    
    # Find all event files recursively across all provided directories
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

def analyze_rollout_ep_true_mean(log_dirs, save_plot=True):
    """
    Analyze the rollout/ep_true_mean metric for different file categories.
    
    Args:
        log_dirs (list): List of paths to directories containing TensorBoard event files
        save_plot (bool): Whether to save the plot to file
    """
    # Extract data with file path information
    data = get_tensorboard_data_with_file_info(log_dirs, tags=['rollout/ep_true_mean'])
    
    if 'rollout/ep_true_mean' not in data:
        print("No rollout/ep_true_mean metric found in the data")
        return
    
    df = data['rollout/ep_true_mean']
    
    # Categorize files
    random_mask = df['file_path'].str.contains('random_')
    active_mask = df['file_path'].str.contains('active_') | df['file_path'].str.contains('bald_')
    
    df_random = df[random_mask].copy()
    df_active = df[active_mask].copy()
    
    print(f"Random files: {df_random['file_path'].nunique()} unique files")
    print(f"Active/BALD files: {df_active['file_path'].nunique()} unique files")
    
    # Calculate statistics for each step
    def calculate_step_statistics(df_category, category_name):
        if df_category.empty:
            return pd.DataFrame()
        stats = df_category.groupby('step')['value'].agg(['mean', 'std', 'count']).reset_index()
        stats['std'] = stats['std'].fillna(0) # Treat std of a single point as 0
        stats['category'] = category_name
        stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
        stats['ci_lower'] = stats['mean'] - 1.96 * stats['stderr']
        stats['ci_upper'] = stats['mean'] + 1.96 * stats['stderr']
        return stats
    
    stats_random = calculate_step_statistics(df_random, 'Random')
    stats_active = calculate_step_statistics(df_active, 'Active/BALD')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot random data
    if not stats_random.empty:
        random_runs = df_random['file_path'].nunique()
        plt.plot(stats_random['step'], stats_random['mean'], 
                 label=f'Random', color='blue', linewidth=2)
        plt.fill_between(stats_random['step'], 
                         stats_random['ci_lower'], 
                         stats_random['ci_upper'], 
                         alpha=0.3, color='blue')
    
    # Plot active/BALD data
    if not stats_active.empty:
        active_runs = df_active['file_path'].nunique()
        plt.plot(stats_active['step'], stats_active['mean'], 
                 label=f'Active/BALD', color='red', linewidth=2)
        plt.fill_between(stats_active['step'], 
                         stats_active['ci_lower'], 
                         stats_active['ci_upper'], 
                         alpha=0.3, color='red')
    
    plt.xlabel('Step')
    plt.ylabel('rollout/ep_true_mean')
    plt.title('Evolution of rollout/ep_true_mean: Random vs Active/BALD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        plt.savefig('rollout_ep_true_mean_comparison.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'rollout_ep_true_mean_comparison.png'")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    if not stats_random.empty:
        print("Random:")
        print(f"  Final mean: {stats_random['mean'].iloc[-1]:.4f}")
        print(f"  Final std: {stats_random['std'].iloc[-1]:.4f}")
        print(f"  Number of runs: {df_random['file_path'].nunique()}")
    else:
        print("Random: No data found.")
    
    if not stats_active.empty:
        print("Active/BALD:")
        print(f"  Final mean: {stats_active['mean'].iloc[-1]:.4f}")
        print(f"  Final std: {stats_active['std'].iloc[-1]:.4f}")
        print(f"  Number of runs: {df_active['file_path'].nunique()}")
    else:
        print("Active/BALD: No data found.")
    
    return stats_random, stats_active

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
    
    args = parser.parse_args()
    
    if args.analyze_rollout:
        analyze_rollout_ep_true_mean(args.log_dirs)
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