import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_correlation_by_bin(clips, reward_model, iteration, results_dir, bin_width=1.0):
    true_sums = [sum(c["rews"]) for c in clips]
    predicted_sums = [
        sum(reward_model.predict_reward(o, a)
            for o, a in zip(c["obs"], c["acts"]))
        for c in clips
    ]
    min_r, max_r = np.floor(min(true_sums)), np.ceil(max(true_sums))
    bins = np.arange(min_r, max_r + bin_width, bin_width)
    bin_idxs = np.digitize(true_sums, bins) - 1

    bin_centers, corrs = [], []
    for i in range(len(bins) - 1):
        idxs = [j for j, b in enumerate(bin_idxs) if b == i]
        if len(idxs) >= 2:
            ts = [true_sums[j]      for j in idxs]
            ps = [predicted_sums[j] for j in idxs]
            r, _ = pearsonr(ts, ps)
        else:
            r = np.nan
        bin_centers.append((bins[i] + bins[i+1]) / 2)
        corrs.append(r)

    plt.figure(figsize=(8,5))
    plt.plot(bin_centers, corrs, marker='o', linestyle='-')
    plt.xlabel('True reward bin (midpoint)')
    plt.ylabel('Pearson r(true vs. pred)')
    plt.title(f'Iteration {iteration}: reward-model correlation by true-reward bin')
    plt.grid(True)
    out_path = os.path.join(results_dir, f"iter_{iteration}_corr_by_bin.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved correlation-by-bin plot to {out_path}")


def plot_bins(bins, bin_edges, results_dir=None, iteration=0):
    sorted_bin_indices = sorted(bins.keys())
    clip_counts = [len(bins[idx]) for idx in sorted_bin_indices]
    tick_labels = []
    num_defined_bins = len(bin_edges) - 1
    for idx in sorted_bin_indices:
        if idx == -1:
            label = "< {:.2f}".format(bin_edges[0])
        elif idx == num_defined_bins:
            label = ">= {:.2f}".format(bin_edges[-1])
        elif 0 <= idx < num_defined_bins:
            label = "[{:.2f}, {:.2f})".format(bin_edges[idx], bin_edges[idx+1])
        else:
            label = f"Bin {idx}" 
        tick_labels.append(label)
    x_positions = np.arange(len(sorted_bin_indices))
    plt.figure(figsize=(max(10, len(tick_labels) * 0.8), 6))
    plt.bar(x_positions, clip_counts, width=0.5, align='center')
    plt.xlabel('Clip Reward Range')
    plt.ylabel('Number of Clips')
    plt.title(f'Number of Clips in Each Bin (Iteration {iteration})')
    plt.xticks(x_positions, tick_labels, rotation=45, ha="right")
    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/bins_{iteration}.png")
    else:
        plt.show()
    plt.close()

def plot_rewards(clip_rewards, results_dir=None, it=0):
    plt.figure(figsize=(10, 6))
    plt.hist(clip_rewards, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Iteration {it} - Distribution of Collected Clip Rewards")
    plt.xlabel("Sum of True Rewards per Clip")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plot_path_clips = os.path.join(results_dir, f"iter_{it}_clip_rewards_dist.png")
    plt.savefig(plot_path_clips)
    plt.close()
    print(f"Saved clip rewards distribution plot to {plot_path_clips}")