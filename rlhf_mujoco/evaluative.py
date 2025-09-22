import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from plots import plot_bins

# Helper to sample from a truncated normal distribution via simple rejection sampling
def _sample_truncated_normal(mu: float, sigma: float, lower: float, upper: float, rng: np.random.Generator | None = None) -> float:
    """Draw one sample from N(mu, sigma^2) truncated to [lower, upper]."""
    if rng is None:
        rng = np.random.default_rng()
    if sigma <= 0:
        # Degenerate case: return clipped mean
        return float(np.clip(mu, lower, upper))
    # Rejection sampling; for the given bounds and sigma in this project, it will be fast
    while True:
        x = rng.normal(mu, sigma)
        if lower <= x <= upper:
            return float(x)

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

def annotate_evaluative(clips, num_bins=10, rating_range=None, gamma=0.99, noisy: bool = False, beta: float = 0.5):
    """
    Annotates clips with evaluative ratings based on discounted returns.
    Uses a fixed rating range for Hopper, otherwise calculates it from the clips.
    """
    evaluative_data = []
    returns = []
    if rating_range is not None:
        min_return, max_return = rating_range
    bin_edges = np.linspace(min_return, max_return, num_bins + 1)
    for clip in clips:
        discounted_return = calculate_return(clip, gamma=gamma)
        print(f"Discounted return for clip: {discounted_return}")
        returns.append(discounted_return)
        clipped_return = np.clip(discounted_return, min_return, max_return)
        print(f"Clipped return: {clipped_return}")
        rating = np.digitize(clipped_return, bin_edges[1:-1])
        print(f"Rating for clip: {rating}")
        if noisy:
            # Add truncated Gaussian noise directly to 0-based bin rating
            baseline = float(rating)
            sigma = beta * 10.0
            noise = _sample_truncated_normal(mu=0.0, sigma=sigma, lower=1.0, upper=10.0)
            noisy_rating = baseline + noise
            evaluative_data.append((clip, float(noisy_rating)))
        else:
            # Keep original 0..num_bins-1 rating without offset
            evaluative_data.append((clip, float(rating)))
    return evaluative_data, returns, bin_edges

def calculate_return(clip, gamma=0.99):
    """
    Calculate the discounted return for a trajectory segment.
    R(ξ) = Σ(t=0 to L-1) gamma^t * r_t
    """
    rewards = clip["rews"]
    return sum(r * (gamma ** i) for i, r in enumerate(rewards))