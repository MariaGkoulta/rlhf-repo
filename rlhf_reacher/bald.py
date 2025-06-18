import torch
import torch.nn.functional as F
import numpy as np
import random

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

def select_active_pairs(clips, model, pool_size=50_000, K=500, T=10, device='cpu'): # Added device parameter
    num_clips_to_sample_for_pairs = pool_size * 2
    if len(clips) < 2:
        print("Warning: Not enough clips to form any pairs for BALD.")
        return []
    if len(clips) < num_clips_to_sample_for_pairs:
        print(f"Requested {num_clips_to_sample_for_pairs} clips for BALD candidate pool, but only {len(clips)} available. Using all available.")
        num_clips_to_sample_for_pairs = len(clips)
    candidates = random.sample(clips, num_clips_to_sample_for_pairs)
    pairs = [(candidates[i], candidates[i+1]) 
             for i in range(0, len(candidates) - (len(candidates) % 2), 2)]
    if not pairs:
        print("Warning: No pairs formed for BALD scoring.")
        return []
    scores = [bald_score(model, c1, c2, T, device=device) for c1,c2 in pairs]
    actual_K = min(K, len(scores))
    if actual_K == 0:
        return []
    idxs = np.argsort(scores)[-actual_K:]
    return [pairs[i] for i in idxs]
