import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp
import math

# NORMALIZATION FLAGS FOR BALD SCORES
PREF_BALD_NORM_BY_LOG2 = True       # divide preference BALD by log(2) to map to [0,1]
EVAL_BALD_NORM_PERCENTILE = 99      # high percentile for evaluative‐BALD normalization
_eval_bald_high = 1e-8              # running high value for evaluative‐BALD

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
        if hasattr(model, 'probabilistic') and model.probabilistic:
            r1_mean, r1_var = model(s1, a1)
            r2_mean, r2_var = model(s2, a2)
            r1 = r1_mean.sum()
            r2 = r2_mean.sum()
        else:
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

    # raw BALD (nats); normalize to bits if requested
    raw = H_mean - E_H
    if PREF_BALD_NORM_BY_LOG2:
        raw = raw / math.log(2)
    return raw

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

def select_active_pairs(clips, model, pool_size=50_000, K=500, T=10, device='cpu', logger=None, iteration=0, results_dir=None, min_gap=0): # Added device parameter & min_gap
    """Select pairs with highest BALD score subject to a minimum return gap.

    We now enforce the min_gap constraint BEFORE scoring so that no pairs are
    later discarded for failing the constraint. This prevents wasted BALD
    computations and ensures exactly the intended number of qualifying pairs
    (up to availability) are considered.
    """
    print(f"Selecting {K} active pairs from {len(clips)} clips with pool size {pool_size}, T={T}, min_gap={min_gap}")
    if len(clips) < 2:
        return []
    # Precompute clip returns for efficiency
    returns = [sum(c["rews"]) for c in clips]
    pair_candidates = set()
    number_of_pairs_to_collect = pool_size * 50  # large over-sampling budget
    max_attempts = pool_size * 200  # allow more attempts when min_gap is restrictive
    attempts = 0
    while len(pair_candidates) < number_of_pairs_to_collect and attempts < max_attempts:
        c1_idx, c2_idx = random.sample(range(len(clips)), 2)
        if c1_idx > c2_idx:
            c1_idx, c2_idx = c2_idx, c1_idx
        # Enforce min_gap here
        if abs(returns[c1_idx] - returns[c2_idx]) >= min_gap:
            pair_candidates.add((c1_idx, c2_idx))
        attempts += 1
    if len(pair_candidates) == 0:
        print(f"Warning: No candidate pairs met min_gap={min_gap}. Returning empty list.")
        return []
    pairs = [(clips[i], clips[j]) for i, j in pair_candidates]
    scores = [bald_score(model, c1, c2, T, device=device) for c1, c2 in pairs]

    if results_dir:
        plot_bald_diagnostics(pairs, scores, results_dir, iteration)

    if logger is not None:
        logger.record("active_learning/avg_bald_score", np.mean(scores))
        logger.record("active_learning/bald_variance", np.var(scores))
        logger.dump(iteration)

    actual_K = min(K, len(scores))
    idxs = np.argsort(scores)[-actual_K:]
    return [pairs[i] for i in idxs]

def get_bald_scores_for_pairs(clips, model, pool_size, T, device, logger=None, iteration=0, min_gap=0):
    """Calculates and returns BALD scores for a pool of candidate pairs respecting min_gap."""
    if len(clips) < 2:
        return [], []
    returns = [sum(c["rews"]) for c in clips]
    pair_candidates = set()
    max_attempts = pool_size * 50  # allow more tries for restrictive gaps
    attempts = 0
    while len(pair_candidates) < pool_size and attempts < max_attempts:
        c1_idx, c2_idx = random.sample(range(len(clips)), 2)
        if c1_idx > c2_idx:
            c1_idx, c2_idx = c2_idx, c1_idx
        if abs(returns[c1_idx] - returns[c2_idx]) >= min_gap:
            pair_candidates.add((c1_idx, c2_idx))
        attempts += 1
    if len(pair_candidates) == 0:
        print(f"Warning: No candidate pairs met min_gap={min_gap}. Returning empty lists.")
        return [], []
    pairs = [(clips[i], clips[j]) for i, j in pair_candidates]
    scores = [bald_score(model, c1, c2, T, device=device) for c1, c2 in pairs]
    order = np.argsort(scores)[::-1]
    pairs = [pairs[i] for i in order]
    scores = [scores[i] for i in order]
    print(f"Average BALD score for pairs (min_gap={min_gap}): {np.mean(scores)} from {len(scores)} candidates (attempts={attempts}).")
    if logger is not None:
        logger.record("active_learning/avg_bald_score", np.mean(scores))
        logger.record("active_learning/bald_variance", np.var(scores))
        logger.dump(iteration)
    return pairs, scores

def mixture_entropy(mus, vars, n_samples=None, device=None):
    if device is None:
        device = mus.device
    M = mus.shape[0]
    comp_idx = torch.randint(0, M, (n_samples,), device=device)
    y = torch.normal(mus[comp_idx], torch.sqrt(vars[comp_idx]))
    y_exp   = y.unsqueeze(1)                 # (n_samples, 1)
    mus_exp = mus.unsqueeze(0)               # (1, M)
    vars_exp = vars.unsqueeze(0)               # (1, M)
    # Analytic log‑pdf of Gaussian
    #   log N(y | μ_j, σ_j^2) = -½[ log(2πσ_j^2) + (y−μ_j)^2/σ_j^2 ]
    log_probs = -0.5 * (torch.log(2 * torch.tensor(math.pi) * vars_exp)
                       + (y_exp - mus_exp)**2 / vars_exp)
    # 4) Compute log p(y_i) under the mixture via log‑sum‑exp
    log_mixture = torch.logsumexp(log_probs, dim=1) - torch.log(torch.tensor(M, device=device))
    # 5) Estimate entropy: –E[ log p(y) ]
    return -log_mixture.mean()

def evaluative_bald_score(model, clip, T=10, device='cpu', rating_range=(0, 10), num_bins=10):
    """
    Calculates the BALD score for a clip for evaluative feedback using the regression formulation.
    BALD(x) = H[E_θ[p(y|x,θ)]] - E_θ[H[p(y|x,θ)]]
    where y is the predicted return (a continuous value).
    This implementation is for a probabilistic model that outputs mean and variance.
    """
    if not (hasattr(model, 'probabilistic') and model.probabilistic):
        raise ValueError("evaluative_bald_score requires a probabilistic model. Please set USE_PROBABILISTIC_MODEL to True in your config.")
    model.train()  # Enable dropout
    s, a = stack_obs_acts(clip, device)
    mus = torch.zeros(T)
    sigmas_sq = torch.zeros(T)
    with torch.no_grad():
        for t in range(T):
            # Get mean and variance for per-step rewards from the model
            per_step_rewards_mean, per_step_rewards_var = model(s, a)
            # Summing means and variances for the undiscounted return of the clip
            # Assumes independence of rewards at each timestep for a given model sample theta_t
            mus[t] = per_step_rewards_mean.sum().item()
            sigmas_sq[t] = per_step_rewards_var.sum().item()
    mean_of_entropies = torch.mean(1/2 * torch.log(2 * torch.pi * torch.e * sigmas_sq))
    bald_score = mixture_entropy(mus, sigmas_sq, n_samples=T, device=device) - mean_of_entropies

    return bald_score

def select_active_clips_for_evaluation(clips, model, K=500, T=10, device='cpu', logger=None, iteration=0, gamma=0.99, rating_range=(0, 10)):
    print(f"Selecting {K} active clips for evaluation from {len(clips)} clips with T={T}")
    if not clips:
        return [], [], []
    if not (hasattr(model, 'probabilistic') and model.probabilistic):
        raise ValueError("select_active_clips_for_evaluation requires a probabilistic model. Please set USE_PROBABILISTIC_MODEL to True in your config.")
    scores = [evaluative_bald_score(model, c, T, device=device, rating_range=rating_range) for c in clips]

    # normalize evaluative BALD scores by running high percentile
    global _eval_bald_high
    p = np.percentile(scores, EVAL_BALD_NORM_PERCENTILE)
    _eval_bald_high = max(_eval_bald_high, p)
    scores = [s / _eval_bald_high for s in scores]

    all_rewards = [sum(c["rews"]) for c in clips]
    if logger is not None:
        logger.record("active_learning/avg_evaluative_bald_score", np.mean(scores))
        logger.record("active_learning/evaluative_bald_variance", np.var(scores))
        logger.dump(iteration)
    actual_K = min(K, len(scores))
    idxs = np.argsort(scores)[-actual_K:]
    selected_clips = [clips[i] for i in idxs]
    selected_rewards = [all_rewards[i] for i in idxs]
    return selected_clips, all_rewards, selected_rewards

def get_bald_scores_for_clips(clips, model, T, device, rating_range, num_samples=None, logger=None, iteration=0):
    """Calculates and returns BALD scores for a list of clips."""
    if num_samples is not None and num_samples < len(clips):
        selected_indices = random.sample(range(len(clips)), num_samples)
        selected_clips = [clips[i] for i in selected_indices]
    else:
        selected_clips = clips
    scores = [evaluative_bald_score(model, c, T, device=device, rating_range=rating_range) for c in selected_clips]

    # normalize evaluative BALD scores by running high percentile
    global _eval_bald_high
    p = np.percentile(scores, EVAL_BALD_NORM_PERCENTILE)
    _eval_bald_high = max(_eval_bald_high, p)
    print(f"Running high percentile for evaluative BALD: {_eval_bald_high}")
    scores = [s / _eval_bald_high for s in scores]

    # sort clips by descending BALD score
    clips = [selected_clips[i] for i in np.argsort(scores)[::-1]]
    scores = sorted(scores, reverse=True)

    print(f"Average BALD score for selected clips: {np.mean(scores)}")
    if logger is not None:
        logger.record("active_learning/avg_evaluative_bald_score", np.mean(scores))
        logger.record("active_learning/evaluative_bald_variance", np.var(scores))
        logger.dump(iteration)
    return clips, scores

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
            if hasattr(model, 'probabilistic') and model.probabilistic:
                r1_mean, r1_var = model(s1, a1)
                r2_mean, r2_var = model(s2, a2)
                r1 = r1_mean.sum()
                r2 = r2_mean.sum()
            else:
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