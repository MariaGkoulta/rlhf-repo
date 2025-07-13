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
        if self.segment_len is None:
            return torch.tensor(obs, dtype=torch.float32, device=self.device), \
                   torch.tensor(acts, dtype=torch.float32, device=self.device)
        if current_len > self.segment_len:
            obs = obs[:self.segment_len]
            acts = acts[:self.segment_len]
        elif current_len < self.segment_len:
            pad_len = self.segment_len - current_len
            obs_pad = np.zeros((pad_len, obs.shape[1]), dtype=np.float32)
            acts_pad = np.zeros((pad_len, acts.shape[1]), dtype=np.float32)
            obs = np.concatenate([obs, obs_pad], axis=0)
            acts = np.concatenate([acts, acts_pad], axis=0)
        return torch.tensor(obs, dtype=torch.float32, device=self.device), \
               torch.tensor(acts, dtype=torch.float32, device=self.device)

    def __getitem__(self, idx):
        return self.s1[idx], self.a1[idx], self.s2[idx], self.a2[idx], self.prefs[idx]

def clip_return(c):
    return sum(c["rews"])

def annotate_pairs(pairs_to_annotate, min_gap=MIN_GAP, beta=None):
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
        if beta is not None:
            prob_c1_preferred = 1 / (1 + np.exp(-beta * (r1 - r2)))
            preference = 1.0 if random.random() < prob_c1_preferred else 0.0
        else:
            preference = 1 if r1 > r2 else 0
        prefs.append((c1, c2, preference))
        reward_differences.append(difference)
        rewards_log.append((r1, r2))
    return prefs, reward_differences, rewards_log
        