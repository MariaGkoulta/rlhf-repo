import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


NUM_PAIRS = 20_000
MIN_GAP = 2
DEFAULT_SAMPLES_PER_OTHER_BIN = 1000
LOWER_BIN = -20
UPPER_BIN = 0

class PreferenceDataset(Dataset):
    """
    Stores preference-labeled clip pairs as tensors, converting each clip only once on addition.
    """
    def __init__(self, device='cpu'):
        self.s1 = []
        self.a1 = []
        self.s2 = []
        self.a2 = []
        self.prefs = []
        self.device = device

    def add(self, clip1, clip2, pref):
        s1 = torch.tensor(np.stack(clip1['obs']), dtype=torch.float32, device=self.device)
        a1 = torch.tensor(np.stack(clip1['acts']), dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack(clip2['obs']), dtype=torch.float32, device=self.device)
        a2 = torch.tensor(np.stack(clip2['acts']), dtype=torch.float32, device=self.device)
        p  = torch.tensor(pref, dtype=torch.float32, device=self.device)
        self.s1.append(s1)
        self.a1.append(a1)
        self.s2.append(s2)
        self.a2.append(a2)
        self.prefs.append(p)

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, idx):
        return self.s1[idx], self.a1[idx], self.s2[idx], self.a2[idx], self.prefs[idx]


def annotate_preferences(clips, num_pairs=NUM_PAIRS, min_gap=MIN_GAP):
    prefs = []
    reward_differences = []
    rewards = []
    for _ in tqdm(range(num_pairs)):
        c1, c2 = random.sample(clips, 2)
        r1, r2 = clip_return(c1), clip_return(c2)
        difference = abs(r1 - r2)
        if abs(r1 - r2) < min_gap:
            continue
        prefs.append((c1, c2, 1 if r1 > r2 else 0))
        reward_differences.append(difference)
        rewards.append((r1, r2))
    return prefs, reward_differences, rewards

def clip_return(c):
    return sum(c["rews"])

def create_bins(bins, clips, results_dir=None, iteration=0, num_bins=10):
    bin_edges = np.linspace(LOWER_BIN, UPPER_BIN, num_bins + 1)
    if bins is None:
        bins_to_fill = defaultdict(list)
    else:
        bins_to_fill = bins
    for clip in clips:
        r = clip_return(clip)
        bin_idx = np.digitize(r, bin_edges) - 1
        bins_to_fill[bin_idx].append(clip)
    plot_bins(bins_to_fill, bin_edges, results_dir=results_dir, iteration=iteration)
    return bins_to_fill

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

def create_preferences(bins, num_samples_per_other_bin=DEFAULT_SAMPLES_PER_OTHER_BIN, min_gap=MIN_GAP):
    """
    Creates preference pairs based on binned clips.
    For each bin, one clip is randomly selected. This clip is then paired with
    `num_samples_per_other_bin` clips randomly selected from each of the other bins.
    """
    prefs = []
    reward_differences = []
    generated_rewards = []
    bin_keys = list(bins.keys())
    
    for bin1 in bin_keys:
        for bin2 in bin_keys:
            if not bins[bin1] or not bins[bin2]:
                continue
            actual_samples_from_target_bin = min(num_samples_per_other_bin, len(bins[bin2]), len(bins[bin1]))
            if actual_samples_from_target_bin == 0:
                continue
            sampled_clips_from_source = random.sample(bins[bin1], actual_samples_from_target_bin)
            sampled_clips_from_target = random.sample(bins[bin2], actual_samples_from_target_bin)
            for clip1, clip2 in zip(sampled_clips_from_source, sampled_clips_from_target):
                r1 = clip_return(clip1)
                r2 = clip_return(clip2)
                difference = abs(r1 - r2)
                if difference < min_gap:
                    continue
                preference_label = 1 if r1 > r2 else 0
                prefs.append((clip1, clip2, preference_label))
                reward_differences.append(difference)
                generated_rewards.append((r1, r2))

    return prefs, reward_differences, generated_rewards


def annotate_given_pairs(pairs_to_annotate, min_gap=MIN_GAP):
    print(f"Number of pairs to annotate: {len(pairs_to_annotate)}")
    prefs = []
    reward_differences = []
    rewards_log = []
    print(f"Annotating {len(pairs_to_annotate)} BALD selected pairs...")
    for c1, c2 in tqdm(pairs_to_annotate, desc="Annotating selected pairs"):
        r1, r2 = clip_return(c1), clip_return(c2)
        difference = abs(r1 - r2)
        if difference < min_gap:
            continue
        prefs.append((c1, c2, 1 if r1 > r2 else 0))
        reward_differences.append(difference)
        rewards_log.append((r1, r2))
    print(f"Annotated {len(prefs)} pairs with preferences.")
    return prefs, reward_differences, rewards_log
        