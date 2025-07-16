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

def bald_score(model, pairs, T=10, device='cpu'):
    """
    Compute BALD scores for a batch of pairs.
    pairs: list of (clip1, clip2)
    Returns: np.array of BALD scores, shape (len(pairs),)
    """
    model.train()
    n = len(pairs)
    s1s_list, a1s_list, s2s_list, a2s_list = [], [], [], []
    for c1, c2 in pairs:
        s1, a1 = stack_obs_acts(c1, device)
        s2, a2 = stack_obs_acts(c2, device)
        s1s_list.append(s1)
        a1s_list.append(a1)
        s2s_list.append(s2)
        a2s_list.append(a2)
    # Pad to max length for batching
    max_len1 = max(s.shape[0] for s in s1s_list) if s1s_list else 0
    max_len2 = max(s.shape[0] for s in s2s_list) if s2s_list else 0
    def pad_seq(seq_list, max_len):
        return torch.stack([
            F.pad(s, (0,0,0,max_len-s.shape[0])) for s in seq_list
        ])
    s1s = pad_seq(s1s_list, max_len1)
    a1s = pad_seq(a1s_list, max_len1)
    s2s = pad_seq(s2s_list, max_len2)
    a2s = pad_seq(a2s_list, max_len2)
    # Mask for valid timesteps
    mask1 = torch.tensor([[1]*s.shape[0] + [0]*(max_len1-s.shape[0]) for s in s1s_list], device=device, dtype=torch.float32)
    mask2 = torch.tensor([[1]*s.shape[0] + [0]*(max_len2-s.shape[0]) for s in s2s_list], device=device, dtype=torch.float32)
    probs = []
    for _ in range(T):
        r1_per_step = model(s1s, a1s)
        r2_per_step = model(s2s, a2s)
        
        r1 = (r1_per_step * mask1).sum(dim=1)
        r2 = (r2_per_step * mask2).sum(dim=1)

        p = torch.sigmoid(r1 - r2)
        probs.append(p.detach().cpu().numpy())
    probs = np.stack(probs, axis=1)  # shape (n, T)
    p_mean = probs.mean(axis=1)
    H_mean = - (p_mean*np.log(p_mean + 1e-8) + (1-p_mean)*np.log(1-p_mean+1e-8))
    H_t = - (probs*np.log(probs+1e-8) + (1-probs)*np.log(1-probs+1e-8))
    E_H = H_t.mean(axis=1)
    return H_mean - E_H

def select_active_pairs(clips, model, pool_size=50_000, K=500, T=10, device='cpu', logger=None, iteration=0, results_dir=None, batch_size=128):
    """
    Selects K pairs of clips with the highest BALD score.
    Now uses batched BALD computation for speed.
    """
    print(f"Selecting {K} active pairs from {len(clips)} clips with pool size {pool_size} and T={T}")
    if len(clips) < 2:
        return []
    num_clips_to_sample = int(np.ceil((1 + np.sqrt(1 + 8 * pool_size)) / 2))
    num_clips_to_sample = min(len(clips), num_clips_to_sample)
    if num_clips_to_sample < 2:
        print("Not enough clips to form a pair after sampling.")
        return []
    print(f"Sampling {num_clips_to_sample} clips to generate pair candidates.")
    candidate_clips = random.sample(clips, num_clips_to_sample)
    pairs = []
    for i in range(len(candidate_clips)):
        for j in range(i + 1, len(candidate_clips)):
            pairs.append((candidate_clips[i], candidate_clips[j]))
    if not pairs:
        print("No pairs were generated.")
        return []

    # Batched BALD scoring
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        batch_scores = bald_score(model, batch, T, device=device)
        scores.extend(batch_scores)
    scores = np.array(scores)

    if results_dir:
        plot_bald_diagnostics(pairs, scores, results_dir, iteration)

    if logger is not None:
        logger.record("active_learning/avg_bald_score", np.mean(scores))
        logger.record("active_learning/bald_variance", np.var(scores))
        logger.dump(iteration)

    actual_K = min(K, len(scores))
    idxs = np.argsort(scores)[-actual_K:]
    return [pairs[i] for i in idxs]

def plot_bald_diagnostics(pairs, scores, results_dir, iteration):
    """
    Visualizes BALD scores to diagnose active learning performance.
    - Plots a histogram of BALD scores.
    - Plots BALD scores vs. the true reward difference of the pairs.
    """
    if not scores or len(scores) == 0:
        print("No scores to plot for BALD diagnostics.")
        return

    true_reward_diffs = [abs(sum(c1['rews']) - sum(c2['rews'])) for c1, c2 in pairs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram of BALD scores
    ax1.hist(scores, bins=50)
    ax1.set_title(f'BALD Score Distribution (Iter {iteration})')
    ax1.set_xlabel('BALD Score')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
    ax1.legend()

    # Scatter plot of BALD score vs. true reward difference
    ax2.scatter(true_reward_diffs, scores, alpha=0.5)
    ax2.set_title(f'BALD Score vs. True Reward Difference (Iter {iteration})')
    ax2.set_xlabel('Absolute True Reward Difference')
    ax2.set_ylabel('BALD Score')

    plt.tight_layout()
    diagnostic_dir = os.path.join(results_dir, "bald_diagnostics")
    os.makedirs(diagnostic_dir, exist_ok=True)
    plt.savefig(os.path.join(diagnostic_dir, f"bald_diagnostic_iter_{iteration}.png"))
    plt.close(fig)

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