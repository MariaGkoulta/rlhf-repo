import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import io

def plot_correlation_by_bin(clips, reward_model, iteration, results_dir, bin_width=1.0, writer=None):
    true_sums = [sum(c["rews"]) for c in clips]
    predicted_sums = []
    for clip in clips:
        total = 0.0
        for o, a in zip(clip["obs"], clip["acts"]):
            pred = reward_model.predict_reward(o, a)
            if isinstance(pred, tuple):
                total += pred[0]
            else:
                total += pred
        predicted_sums.append(total)
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

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(bin_centers, corrs, marker='o', linestyle='-')
    ax.set_xlabel('True reward bin (midpoint)')
    ax.set_ylabel('Pearson r(true vs. pred)')
    ax.set_title(f'Iteration {iteration}: reward-model correlation by true-reward bin')
    ax.grid(True)
    out_path = os.path.join(results_dir, f"iter_{iteration}_corr_by_bin.png")
    fig.tight_layout()
    fig.savefig(out_path)
    if writer:
        log_plot_to_tensorboard(fig, writer, out_path, iteration)
    plt.close(fig)
    print(f"Saved correlation-by-bin plot to {out_path}")

def plot_preference_heatmap(reward_pairs, results_dir, iteration, num_bins=12, range_min=-60, range_max=0):
    """
    Generates and saves a heatmap of preference pair rewards.

    Args:
        reward_pairs: A list of tuples, where each tuple is (r1, r2) for a preference pair.
        results_dir: The directory to save the plot.
        iteration: The current training iteration number.
        num_bins: The number of bins for each axis of the heatmap.
        range_min: The minimum reward value for the axes.
        range_max: The maximum reward value for the axes.
    """
    if not reward_pairs:
        print("No reward pairs to plot for heatmap.")
        return
    r1s = [pair[0] for pair in reward_pairs]
    r2s = [pair[1] for pair in reward_pairs]
    fig, ax = plt.subplots(figsize=(8, 7))
    h, xedges, yedges, image = ax.hist2d(
        r1s, r2s, 
        bins=num_bins, 
        range=[[range_min, range_max], [range_min, range_max]],
        cmap='viridis',
        cmin=1
    )
    fig.colorbar(image, ax=ax, label='Number of Pairs')
    ax.set_title(f'Preference Pair Reward Distribution - Iteration {iteration}')
    ax.set_xlabel('Return of Clip 1')
    ax.set_ylabel('Return of Clip 2')
    ax.set_aspect('equal', adjustable='box')
    if results_dir:
        plot_path = os.path.join(results_dir, f'preference_heatmap_iter_{iteration}.png')
        plt.savefig(plot_path)
        print(f"Saved preference heatmap to {plot_path}")
    
    plt.close(fig)


def plot_bins(bins, bin_edges, results_dir=None, iteration=0, writer=None):
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
    plt.close()

def plot_rewards(clip_rewards, results_dir=None, it=0, writer=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clip_rewards, bins=50, color='skyblue', edgecolor='black')
    ax.set_title(f"Iteration {it} - Distribution of Collected Clip Rewards")
    ax.set_xlabel("Sum of True Rewards per Clip")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.75)
    plot_path_clips = os.path.join(results_dir, f"iter_{it}_clip_rewards_dist.png")
    fig.savefig(plot_path_clips)
    if writer:
        log_plot_to_tensorboard(fig, writer, plot_path_clips, it)
    plt.close(fig)
    print(f"Saved clip rewards distribution plot to {plot_path_clips}")


def log_plot_to_tensorboard(fig, writer, tag, step):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    import PIL.Image
    image = PIL.Image.open(buf)
    image = np.array(image)
    # TensorBoard expects CHW format and float32
    image = image.transpose(2, 0, 1)[None] / 255.0
    writer.add_image(tag, image[0], step)
    buf.close()


def plot_true_vs_pred(true_rewards, pred_rewards, results_dir, iteration, writer=None):
    """
    Plots a scatter plot of true vs predicted rewards for an iteration.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_rewards, pred_rewards, alpha=0.6, color='purple')
    ax.set_xlabel("True Episode Reward")
    ax.set_ylabel("Predicted Episode Reward")
    ax.set_title(f"Iteration {iteration}: True vs Predicted Episode Reward")
    ax.grid(True)
    ax.plot([min(true_rewards), max(true_rewards)], [min(pred_rewards), max(pred_rewards)], 'r--', label='y=x')
    ax.legend()
    out_path = os.path.join(results_dir, f"iter_{iteration}_true_vs_pred.png")
    fig.tight_layout()
    fig.savefig(out_path)
    if writer:
        log_plot_to_tensorboard(fig, writer, out_path, iteration)
    plt.close(fig)
    print(f"Saved true vs predicted reward scatter plot to {out_path}")