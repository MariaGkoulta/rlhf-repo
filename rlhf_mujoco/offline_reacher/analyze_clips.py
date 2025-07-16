import argparse
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt


def load_and_filter_clips(path):
    with open(path, 'rb') as f:
        clips = pickle.load(f)

    valid_clips = []
    for i, clip in enumerate(clips):
        if not isinstance(clip, dict):
            continue
        obs = clip.get('observations')
        acts = clip.get('actions')
        rew = clip.get('reward')
        if rew is None or not isinstance(obs, np.ndarray) or not isinstance(acts, np.ndarray):
            continue
        if obs.ndim != 2 or acts.ndim != 2:
            continue
        if obs.shape[0] == 0 or obs.shape[0] != acts.shape[0]:
            continue
        valid_clips.append(clip)

    return valid_clips


def compute_descriptive_stats(data):
    """
    Compute basic descriptive statistics for a 1D array.
    """
    return {
        'count': len(data),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        '25%': float(np.percentile(data, 25)),
        '50%': float(np.median(data)),
        '75%': float(np.percentile(data, 75)),
        'max': float(np.max(data))
    }


def plot_histogram(data, title, xlabel, bins=50, save_path=None):
    """
    Plot a histogram of the data.
    """
    plt.figure()
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distributions of clip rewards, lengths, and their pairwise differences"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='collected_clips_with_rewards.pkl',
        help='Path to the pickle file containing clips'
    )
    parser.add_argument(
        '--num_pairs', '-n',
        type=int,
        default=20000,
        help='Number of random pairs to generate'
    )
    parser.add_argument(
        '--bins', '-b',
        type=int,
        default=50,
        help='Number of bins for histograms'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='.',
        help='Directory to save histogram plots'
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    clips = load_and_filter_clips(args.input)
    if len(clips) == 0:
        raise ValueError("No valid clips found after filtering.")

    # Extract rewards and lengths
    rewards = np.array([c['reward'] for c in clips], dtype=float)
    lengths = np.array([c['observations'].shape[0] for c in clips], dtype=int)

    # Descriptive stats
    rewards_stats = compute_descriptive_stats(rewards)
    lengths_stats = compute_descriptive_stats(lengths)

    print("Reward Distribution Stats:")
    for k, v in rewards_stats.items():
        print(f"  {k}: {v}")

    print("\nLength Distribution Stats:")
    for k, v in lengths_stats.items():
        print(f"  {k}: {v}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_histogram(
        rewards,
        title="Histogram of Clip Rewards",
        xlabel="Reward",
        bins=args.bins,
        save_path=os.path.join(args.output_dir, 'rewards_hist.png')
    )
    plot_histogram(
        lengths,
        title="Histogram of Clip Lengths",
        xlabel="Length (timesteps)",
        bins=args.bins,
        save_path=os.path.join(args.output_dir, 'lengths_hist.png')
    )

    # Generate random pairs and compute differences
    num_pairs = args.num_pairs
    N = len(clips)
    idx1 = np.random.randint(0, N, size=num_pairs)
    idx2 = np.random.randint(0, N, size=num_pairs)

    reward_diffs = rewards[idx1] - rewards[idx2]
    length_diffs = lengths[idx1] - lengths[idx2]

    reward_diffs_stats = compute_descriptive_stats(reward_diffs)
    length_diffs_stats = compute_descriptive_stats(length_diffs)

    print("\nReward Difference Distribution Stats (random pairs):")
    for k, v in reward_diffs_stats.items():
        print(f"  {k}: {v}")

    print("\nLength Difference Distribution Stats (random pairs):")
    for k, v in length_diffs_stats.items():
        print(f"  {k}: {v}")

    plot_histogram(
        reward_diffs,
        title="Histogram of Reward Differences (random pairs)",
        xlabel="Reward Difference",
        bins=args.bins,
        save_path=os.path.join(args.output_dir, 'reward_diffs_hist.png')
    )
    plot_histogram(
        length_diffs,
        title="Histogram of Length Differences (random pairs)",
        xlabel="Length Difference (timesteps)",
        bins=args.bins,
        save_path=os.path.join(args.output_dir, 'length_diffs_hist.png')
    )
    print(f"\nPlots saved to directory: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()
