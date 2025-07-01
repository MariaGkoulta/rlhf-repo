import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def stack_obs_acts(clip, device='cpu'):
    obs = torch.tensor(np.stack(clip['obs']), dtype=torch.float32, device=device)
    acts = torch.tensor(np.stack(clip['acts']), dtype=torch.float32, device=device)
    return obs, acts

def bald_score(model, clip1, clip2, T=10, device='cpu'):
    model.train()
    probs = []
    s1, a1 = stack_obs_acts(clip1, device)
    s2, a2 = stack_obs_acts(clip2, device)
    for _ in range(T):
        r1 = model(s1, a1).sum()
        r2 = model(s2, a2).sum()
        p  = torch.sigmoid(r1 - r2)
        probs.append(p.item())
    probs = np.array(probs)
    p_mean = probs.mean()
    # predictive entropy
    H_mean = - (p_mean*np.log(p_mean + 1e-8) + (1-p_mean)*np.log(1-p_mean+1e-8))
    # expected entropy
    H_t = - (probs*np.log(probs+1e-8) + (1-probs)*np.log(1-probs+1e-8))
    E_H   = H_t.mean()
    return H_mean - E_H

def sample_random_pairs(clips, num_pairs, min_gap):
    """Samples pairs of clips randomly, returning the pairs themselves."""
    cand_pairs = []
    if not clips or len(clips) < 2:
        return []
    max_attempts = num_pairs * 100
    attempts = 0
    while len(cand_pairs) < num_pairs and attempts < max_attempts:
        c1, c2 = random.sample(clips, 2)
        if abs(sum(c1["rews"]) - sum(c2["rews"])) >= min_gap:
            cand_pairs.append((c1, c2))
        attempts += 1
    return cand_pairs

def select_active_pairs(clips, model, pool_size=50_000, K=500, T=10, device='cpu', logger=None, iteration=0): # Added device parameter
    if len(clips) < 2:
        return []
    pairs = []
    pair_candidates = set()
    max_attempts = pool_size * 50
    attempts = 0
    while len(pair_candidates) < pool_size and attempts < max_attempts:
        c1_idx, c2_idx = random.sample(range(len(clips)), 2)
        if c1_idx > c2_idx: c1_idx, c2_idx = c2_idx, c1_idx
        pair_candidates.add((c1_idx, c2_idx))
        attempts += 1
    pairs = [(clips[i], clips[j]) for i, j in pair_candidates]
    scores = [bald_score(model, c1, c2, T, device=device) for c1,c2 in pairs]

    if logger is not None:
        logger.record("reward_model/avg_bald_score", np.mean(scores), exclude=("stdout",))
        logger.record("reward_model/max_bald_score", np.max(scores), exclude=("stdout",))
        logger.record("reward_model/min_bald_score", np.min(scores), exclude=("stdout")),
        logger.record("reward_model/bald_variance", np.var(scores), exclude=("stdout",))                                                       
        logger.dump(iteration)

    actual_K = min(K, len(scores))
    idxs = np.argsort(scores)[-actual_K:]
    return [pairs[i] for i in idxs]


def variance_score(model, clip1, clip2, T=10, device='cpu'):
    """Calculates the variance of the predicted reward difference between two clips."""
    model.train()  # Enable dropout for stochastic forward passes
    reward_diffs = []
    s1, a1 = stack_obs_acts(clip1, device)
    s2, a2 = stack_obs_acts(clip2, device)
    with torch.no_grad():
        for _ in range(T):
            r1 = model(s1, a1).sum()
            r2 = model(s2, a2).sum()
            reward_diffs.append((r1 - r2).item())
    return np.var(reward_diffs)

def select_variance_pairs(clips, model, pool_size=50_000, K=500, T=10, device='cpu', results_dir=None, iteration=0):
    """Selects pairs of clips with the highest variance in their predicted reward difference."""
    num_clips_to_sample = min(len(clips), pool_size * 2)
    if num_clips_to_sample < 2:
        print("Not enough clips to form pairs for variance selection.")
        return []
    candidates = random.sample(clips, num_clips_to_sample)
    pairs = [(candidates[i], candidates[i+1]) 
             for i in range(0, len(candidates) - (len(candidates) % 2), 2)]
    if not pairs:
        return []
    scores = [variance_score(model, c1, c2, T, device=device) for c1, c2 in pairs]

    actual_K = min(K, len(scores))
    if actual_K == 0:
        return []
    idxs = np.argsort(scores)[-actual_K:]
    return [pairs[i] for i in idxs]