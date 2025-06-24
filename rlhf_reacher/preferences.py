import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from plots import plot_bins

NUM_PAIRS = 2000
MIN_GAP = 2
DEFAULT_SAMPLES_PER_OTHER_BIN = 20
LOWER_BIN = 0
UPPER_BIN = 150
MAX_BUFFER_LEN = 50000

class PreferenceDataset(Dataset):
    """
    Stores preference-labeled clip pairs as tensors, converting each clip only once on addition.
    """
    def __init__(self, maxlen=MAX_BUFFER_LEN, device='cpu', segment_len=None):
        self.s1 = []
        self.a1 = []
        self.s2 = []
        self.a2 = []
        self.prefs = []
        self.maxlen = maxlen
        self.device = device
        self.segment_len = segment_len

    def add(self, clip1, clip2, pref):
        if len(self.prefs) >= self.maxlen:
            self.s1.pop(0)
            self.a1.pop(0)
            self.s2.pop(0)
            self.a2.pop(0)
            self.prefs.pop(0)
        s1, a1 = self._process_clip(clip1)
        s2, a2 = self._process_clip(clip2)
        p  = torch.tensor(pref, dtype=torch.float32, device=self.device)
        self.s1.append(s1)
        self.a1.append(a1)
        self.s2.append(s2)
        self.a2.append(a2)
        self.prefs.append(p)

    def __len__(self):
        return len(self.prefs)
    
    def _process_clip(self, clip):
        """Pads or truncates a clip to self.segment_len."""
        obs = np.stack(clip['obs'])
        acts = np.stack(clip['acts'])
        
        current_len = obs.shape[0]

        # If segment_len is not set, we don't pad or truncate.
        if self.segment_len is None:
            return torch.tensor(obs, dtype=torch.float32, device=self.device), \
                   torch.tensor(acts, dtype=torch.float32, device=self.device)

        if current_len > self.segment_len:
            # Truncate the clip if it's too long
            obs = obs[:self.segment_len]
            acts = acts[:self.segment_len]
        elif current_len < self.segment_len:
            # Pad the clip with zeros if it's too short
            pad_len = self.segment_len - current_len
            obs_pad = np.zeros((pad_len, obs.shape[1]), dtype=np.float32)
            acts_pad = np.zeros((pad_len, acts.shape[1]), dtype=np.float32)
            obs = np.concatenate([obs, obs_pad], axis=0)
            acts = np.concatenate([acts, acts_pad], axis=0)
        
        return torch.tensor(obs, dtype=torch.float32, device=self.device), \
               torch.tensor(acts, dtype=torch.float32, device=self.device)

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
                preference_label = 1 if r1 > r2 else 0
                prefs.append((clip1, clip2, preference_label))
                reward_differences.append(difference)
                generated_rewards.append((r1, r2))

    return prefs, reward_differences, generated_rewards


def annotate_pairs(pairs_to_annotate, min_gap=MIN_GAP):
    print(f"Number of pairs to annotate: {len(pairs_to_annotate)}")
    prefs = []
    reward_differences = []
    rewards_log = []
    print(f"Annotating {len(pairs_to_annotate)} selected pairs...")
    for c1, c2 in tqdm(pairs_to_annotate, desc="Annotating selected pairs"):
        r1, r2 = clip_return(c1), clip_return(c2)
        difference = abs(r1 - r2)
        if difference < min_gap:
            continue
        prefs.append((c1, c2, 1 if r1 > r2 else 0))
        reward_differences.append(difference)
        rewards_log.append((r1, r2))
    return prefs, reward_differences, rewards_log
        