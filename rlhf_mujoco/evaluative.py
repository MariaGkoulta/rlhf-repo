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

class EvaluativeDataset(Dataset):
    """
    Stores evaluative feedback data as tensors for rating-based training.
    """
    def __init__(self, maxlen=MAX_BUFFER_LEN, device='cpu', segment_len=None):
        self.states = []
        self.actions = []
        self.ratings = []
        self.maxlen = maxlen
        self.device = device
        self.segment_len = segment_len

    def add(self, clip, rating):
        if len(self.ratings) >= self.maxlen:
            self.states.pop(0)
            self.actions.pop(0)
            self.ratings.pop(0)
        
        s, a = self._process_clip(clip)
        r = torch.tensor(rating, dtype=torch.float32, device=self.device)
        self.states.append(s)
        self.actions.append(a)
        self.ratings.append(r)

    def __len__(self):
        return len(self.ratings)
    
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
        return self.states[idx], self.actions[idx], self.ratings[idx]

def annotate_evaluative(clips, num_bins=10, discount_factor=0.99):
    """
    Annotates clips with evaluative ratings based on discounted returns.
    """
    evaluative_data = []
    discounted_returns = []
    for clip in clips:
        discounted_return = calculate_discounted_return(clip, discount_factor)
        discounted_returns.append(discounted_return)
        min_return = min(discounted_returns)
    max_return = max(discounted_returns)
    bin_edges = np.linspace(min_return, max_return, num_bins + 1)
    for clip, discounted_return in zip(clips, discounted_returns):
        bin_idx = np.digitize(discounted_return, bin_edges)
        rating = max(1, min(num_bins, bin_idx))
        evaluative_data.append((clip, rating))
    return evaluative_data, discounted_returns, bin_edges

def calculate_discounted_return(clip, discount_factor=0.99):
    """
    Calculate the discounted return for a trajectory segment.
    R(ξ) = Σ(t=0 to L-1) γ^t * r_t
    """
    rewards = clip["rews"]
    discounted_return = 0.0
    for t, reward in enumerate(rewards):
        discounted_return += (discount_factor ** t) * reward
    return discounted_return