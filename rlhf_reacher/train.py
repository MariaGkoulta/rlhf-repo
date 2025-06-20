import argparse
import os
import datetime
import random
import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from gymnasium.wrappers import TimeLimit
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from preferences import (
    PreferenceDataset, annotate_preferences, clip_return,
    create_bins, create_preferences, annotate_given_pairs
)
from reward import RewardModel
from bald import select_active_pairs, select_variance_pairs
from plots import plot_correlation_by_bin, plot_rewards, plot_true_vs_pred

from torch.utils.tensorboard import SummaryWriter


class LearnedRewardEnv(gym.Wrapper):
    def __init__(self, env, reward_model, normalize_rewards=True):
        super().__init__(env)
        self.reward_model = reward_model
        self.prev_obs = None 
        self._true_sum = 0.0
        self._learned_sum = 0.0 
        self.ep_true = []
        self.ep_learned = []

        # Variables for running normalization of raw learned rewards
        self.raw_learned_r_count = 0
        self.raw_learned_r_mean = 0.0
        self.raw_learned_r_M2 = 0.0 # M2 is the sum of squares of differences from the current mean
        self.reward_norm_epsilon = 1e-8 # Epsilon for reward normalization stddev
        self.normalize_rewards = normalize_rewards # Flag to enable/disable reward normalization

    def get_and_clear_episode_rewards(self):
        """Returns the collected episode rewards and clears the internal lists."""
        true = self.ep_true
        learned = self.ep_learned
        self.ep_true = []
        self.ep_learned = []
        return true, learned

    def _update_running_stats(self, raw_value: float): # Takes raw value from reward model
        self.raw_learned_r_count += 1
        delta = raw_value - self.raw_learned_r_mean
        self.raw_learned_r_mean += delta / self.raw_learned_r_count
        delta2 = raw_value - self.raw_learned_r_mean 
        self.raw_learned_r_M2 += delta * delta2
    
    def _get_running_std(self) -> float:
        """Calculates running standard deviation."""
        if self.raw_learned_r_count < 2:
            return 0.0 
        variance = self.raw_learned_r_M2 / self.raw_learned_r_count # This is population variance if count is N, sample if N-1
        return np.sqrt(variance)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_obs = obs
        self._true_sum = 0.0
        self._learned_sum = 0.0
        return obs, info

    def step(self, action):
        obs, true_r, terminated, truncated, info = super().step(action)
        learned_r_raw = self.reward_model.predict_reward(self.prev_obs, action)

        self._true_sum += true_r
        
        reward_for_ppo = learned_r_raw 

        if self.normalize_rewards:
            self._update_running_stats(learned_r_raw)
            current_std = self._get_running_std()
            if current_std > self.reward_norm_epsilon:
                normalized_learned_r = (learned_r_raw - self.raw_learned_r_mean) / current_std
            else:
                normalized_learned_r = learned_r_raw - self.raw_learned_r_mean
            reward_for_ppo = normalized_learned_r
        
        self._learned_sum += reward_for_ppo 

        done = terminated or truncated
        
        if done:
            info = info.copy()
            info["episode"] = {
                "r_true": self._true_sum,
                "r_learned": self._learned_sum
            }
            self.ep_true.append(self._true_sum)
            self.ep_learned.append(self._learned_sum)

        self.prev_obs = obs
        return obs, reward_for_ppo, terminated, truncated, info

def train_reward_model_batched(
    rm, pref_dataset, batch_size=64, epochs=20,
    val_frac=0.1, patience=10, optimizer=None, device='cpu', regularization_weight=1e-4, 
    logger=None, iteration=0
):
    rm.to(device)
    total = len(pref_dataset)
    val_size = int(total * val_frac)
    train_size = total - val_size
    train_ds, val_ds = random_split(pref_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs+1):
        iteration += 1
        rm.train()
        train_losses, train_accs = [], []
        for s1, a1, s2, a2, prefs in train_loader:
            N, T, obs_dim = s1.shape
            _, _, act_dim = a1.shape

            s1f = s1.view(N*T, obs_dim)
            a1f = a1.view(N*T, act_dim)
            s2f = s2.view(N*T, obs_dim)
            a2f = a2.view(N*T, act_dim)

            r1 = rm(s1f, a1f).view(N, T).sum(1)
            r2 = rm(s2f, a2f).view(N, T).sum(1)
            logits = r1 - r2

            loss = F.binary_cross_entropy_with_logits(logits, prefs)
            loss += regularization_weight * (r1.pow(2).mean() + r2.pow(2).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(((logits>0).float()==prefs).float().mean().item())

        avg_train_loss = float(np.mean(train_losses))
        avg_train_acc = float(np.mean(train_accs))
        print(f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | train_acc={avg_train_acc:.4f}", end=" | ")

        rm.train()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for s1, a1, s2, a2, prefs in val_loader:
                N, T, obs_dim = s1.shape
                s1f = s1.view(N*T, obs_dim)
                a1f = a1.view(N*T, a1.shape[2])
                s2f = s2.view(N*T, obs_dim)
                a2f = a2.view(N*T, a2.shape[2])

                r1 = rm(s1f, a1f).view(N, T).sum(1)
                r2 = rm(s2f, a2f).view(N, T).sum(1)
                logits = r1 - r2

                vloss = F.binary_cross_entropy_with_logits(logits, prefs)
                val_losses.append(vloss.item())
                val_accs.append(((logits>0).float()==prefs).float().mean().item())
        avg_val_acc = float(np.mean(val_accs))
        avg_val_loss = float(np.mean(val_losses))
        print(f"val_loss={avg_val_loss:.4f} | val_acc={avg_val_acc:.4f}")

        if logger is not None:
            logger.record("reward_model/train_loss", avg_train_loss, exclude=("stdout",))
            logger.record("reward_model/train_acc", avg_train_acc, exclude=("stdout",))
            logger.record("reward_model/val_loss", avg_val_loss, exclude=("stdout",))
            logger.record("reward_model/val_acc", avg_val_acc, exclude=("stdout",))
            logger.dump(iteration)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    rm.eval()
    rm.to('cpu')
    return rm, iteration

def collect_clips(policy, num_episodes_to_collect, env_id="Reacher-v4", n_envs=8, max_episode_steps=50): # Renamed num_clips to num_episodes_to_collect
    def make_env(): return gym.make(env_id, render_mode=None, max_episode_steps=max_episode_steps)
    vec_env = DummyVecEnv([make_env]*n_envs)
    obs = vec_env.reset()
    trajs = [{"obs":[], "acts":[], "rews":[]} for _ in range(n_envs)]
    collected_episodes = []
    pbar = tqdm(total=num_episodes_to_collect, desc="Collecting full episodes") 
    while len(collected_episodes) < num_episodes_to_collect:
        actions, _ = policy.predict(obs, deterministic=False)
        next_obs, rewards, dones, infos = vec_env.step(actions)
        for i in range(n_envs):
            trajs[i]["obs"].append(obs[i])
            trajs[i]["acts"].append(actions[i].reshape(-1))
            trajs[i]["rews"].append(rewards[i])
            if dones[i]:
                full_episode_data = trajs[i]
                collected_episodes.append(full_episode_data)
                pbar.update(1)
                if len(collected_episodes) >= num_episodes_to_collect:
                    break
                trajs[i] = {"obs":[], "acts":[], "rews":[]}
        if len(collected_episodes) >= num_episodes_to_collect: 
            break
        obs = next_obs
    pbar.close()
    vec_env.close()
    return collected_episodes[:num_episodes_to_collect]

def extract_segments_from_episodes(
    episodes, 
    segment_len, 
    extract_multiple_segments, 
    target_num_segments_if_multiple=None
):
    """
    Extracts segments from a list of full episodes.

    Args:
        episodes: A list of full trajectories (dictionaries with "obs", "acts", "rews").
        segment_len: The desired length for output segments.
        extract_multiple_segments: Boolean flag. If True, sample target_num_segments_if_multiple.
                                   If False, sample one segment per sufficiently long episode.
        target_num_segments_if_multiple: Integer, only used if extract_multiple_segments is True.

    Returns:
        A list of segments.
    """
    output_segments = []
    eligible_episodes_for_multiple_extraction = [
        ep for ep in episodes if len(ep["acts"]) >= segment_len
    ]
    if extract_multiple_segments:
        if not eligible_episodes_for_multiple_extraction or target_num_segments_if_multiple is None or target_num_segments_if_multiple == 0:
            return []
        for _ in range(target_num_segments_if_multiple):
            selected_episode = random.choice(eligible_episodes_for_multiple_extraction)
            start_index = random.randint(0, len(selected_episode["acts"]) - segment_len)    
            segment = {
                key: val[start_index : start_index + segment_len]
                for key, val in selected_episode.items()
            }
            output_segments.append(segment)
    else:
        for episode in episodes:
            if len(episode["acts"]) >= segment_len:
                start_index = random.randint(0, len(episode["acts"]) - segment_len)
                segment = {
                    key: val[start_index : start_index + segment_len]
                    for key, val in episode.items()
                }
                output_segments.append(segment)
    # print number of segments extracted
    print(f"Extracted {len(output_segments)} segments of length {segment_len} from {len(episodes)} episodes.")
    return output_segments

def sample_random_preferences(clips, num_samples, min_gap):
    cand_pairs = []
    while len(cand_pairs) < num_samples:
        c1, c2 = random.sample(clips, 2)
        if abs(sum(c1["rews"]) - sum(c2["rews"])) >= min_gap:
            cand_pairs.append((c1, c2))
    prefs, _, _ = annotate_given_pairs(cand_pairs, min_gap=min_gap)
    return prefs    

def main():
    TOTAL_ITERS        = 50
    INITIAL_POLICY_TS  = 1

    USE_BINNING        = False  # Set to False to use random sampling instead
    NORMALIZE_REWARDS = True

    EXTRACT_SEGMENTS = True  # If True, segments are extracted from clips
    SEGMENT_LEN       = 30  # Length of segments to extract from clips
    NUM_EPISODES_TO_COLLECT_INITIAL = 200 # Number of full episodes to collect initially.
    NUM_EPISODES_TO_COLLECT_PER_UPDATE = 200  # Number of full episodes to collect in each iteration.
    # Only used if EXTRACT_SEGMENTS is True:
    TARGET_NUM_SEGMENTS_IF_EXTRACTING_INITIAL = 1000 # Target number of segments to sample from initial episodes.
    TARGET_NUM_SEGMENTS_IF_EXTRACTING_PER_UPDATE = 200

    INITIAL_MIN_GAP            = 2
    FINAL_MIN_GAP              = 0

    NUM_BINS           = 120

    BALD_POOL_SIZE     = 50000
    BALD_K             = 10000
    BALD_T             = 10

    REGULARIZATION_WEIGHT = 1e-5
    ENT_COEF           = 0.01  # Entropy coefficient for PPO

    TOTAL_TARGET_PAIRS = 7000
    INITIAL_COLLECTION_FRACTION = 0.25
    PPO_TIMESTEPS_PER_ITER = 20000  # Train policy more often with fewer steps
    REFERENCE_TIMESTEPS_FOR_RATE = 5e6
    # Scales the number of pairs collected per iteration based on the rate.
    # If T_cumulative is 0, rate factor is 1, so this is the initial target pairs per iter in main loop.
    BASE_PAIRS_PER_ITERATION_SCALER = 50

    TOTAL_PPO_TIMESTEPS = 10e6

    MAX_EPISODE_STEPS = 50
    OPTIMIZER_LR = 1e-3
    OPTIMIZER_WD = 1e-4

    env_id = "Reacher-v4"

    parser = argparse.ArgumentParser(
        description=f"Train PPO on {env_id} with random or BALD-based preference sampling"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--random", action="store_true",
                       help="sample preference pairs uniformly at random")
    group.add_argument("--active", action="store_true",
                       help="sample preference pairs via BALD active learning")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    tensorboard_log_dir = f"./logs/ppo_{env_id}/"
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pref_ds = PreferenceDataset(device=device)

    raw_env = gym.make(env_id, render_mode="rgb_array")
    wrapped_base_raw_env = NoSeedArgumentWrapper(raw_env)
    time_limited_raw_env = TimeLimit(wrapped_base_raw_env, max_episode_steps=MAX_EPISODE_STEPS)
    policy = PPO("MlpPolicy", time_limited_raw_env,
                 verbose=1, n_steps=2048, batch_size=64, ent_coef=ENT_COEF, n_epochs=10,
                 tensorboard_log=f"./logs/ppo_{env_id}/")
    policy.learn(total_timesteps=INITIAL_POLICY_TS)
    time_limited_raw_env.close()

    initial_full_episodes = collect_clips(policy, NUM_EPISODES_TO_COLLECT_INITIAL, env_id=env_id, max_episode_steps=MAX_EPISODE_STEPS)
    clips_ds = extract_segments_from_episodes(
        initial_full_episodes,
        SEGMENT_LEN,
        EXTRACT_SEGMENTS,
        TARGET_NUM_SEGMENTS_IF_EXTRACTING_INITIAL if EXTRACT_SEGMENTS else None
    )
    print(f"Collected {len(initial_full_episodes)} initial full episodes.")
    for i, episode_data in enumerate(initial_full_episodes):
        print(f"  Initial Episode {i+1} length: {len(episode_data['acts'])}")

    initial_target_pairs = int(TOTAL_TARGET_PAIRS * INITIAL_COLLECTION_FRACTION)
    print(f"Targeting {initial_target_pairs} initial preference pairs.")

    current_min_gap = INITIAL_MIN_GAP
    print(f"Initial preference generation with MIN_GAP: {current_min_gap}")

    clip_rewards = [clip_return(c) for c in clips_ds]
    plot_rewards(clip_rewards, results_dir, it=0, writer=writer)

    bins = None
    if USE_BINNING:
        bins = create_bins(None, clips_ds, results_dir, 0, NUM_BINS)
        if (NUM_BINS * (NUM_BINS - 1)) > 0: 
            num_samples_per_other_bin_initial = math.ceil(initial_target_pairs / (NUM_BINS * (NUM_BINS - 1)))
            num_samples_per_other_bin_initial = max(1, num_samples_per_other_bin_initial) 
        else: 
            num_samples_per_other_bin_initial = 0 

        if num_samples_per_other_bin_initial > 0:
            print(f"Initial collection (binned): using num_samples_per_other_bin_initial = {num_samples_per_other_bin_initial} to target {initial_target_pairs} pairs.")
            _prefs_list, _, _ = create_preferences(
                bins,
                num_samples_per_other_bin=num_samples_per_other_bin_initial,
                min_gap=current_min_gap
            )
            prefs = _prefs_list
        else:
            print(f"Initial collection (binned): num_samples_per_other_bin_initial is {num_samples_per_other_bin_initial}. No preferences will be generated via create_preferences.")
    else:
        num_rand_initial = initial_target_pairs
        print(f"Initial collection (random): targeting num_rand_initial = {num_rand_initial}")
        if len(clips_ds) >= 2: 
             _prefs_list = sample_random_preferences(clips_ds, num_rand_initial, current_min_gap) # Use calculated value
             prefs = _prefs_list
        else:
            print(f"Initial collection (random): clips_ds has fewer than 2 segments ({len(clips_ds)}). Cannot sample pairs.")

    print(f"Generated {len(prefs)} initial preference pairs.")
    for c1, c2, p in prefs:
        pref_ds.add(c1, c2, p)

    initial_prefs_count = len(pref_ds) # Number of pairs after initial collection
    target_pairs_in_loop = TOTAL_TARGET_PAIRS - initial_prefs_count

    obs_dim = policy.observation_space.shape[0]
    act_dim = policy.action_space.shape[0]
    reward_logger_iteration = 0
    reward_model = RewardModel(obs_dim, act_dim)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=OPTIMIZER_LR, weight_decay=OPTIMIZER_WD)
    reward_model, reward_logger_iteration = train_reward_model_batched(
        reward_model, pref_ds, device=device, epochs=50, patience=10, optimizer=optimizer,
        regularization_weight=REGULARIZATION_WEIGHT, logger=policy.logger, iteration=reward_logger_iteration
    )

    def make_wrapped():
        base_env = gym.make(env_id, render_mode="rgb_array")
        wrapped_base_env = NoSeedArgumentWrapper(base_env)
        e_time_limited = TimeLimit(wrapped_base_env, max_episode_steps=MAX_EPISODE_STEPS)
        e_monitored = Monitor(e_time_limited)
        return LearnedRewardEnv(e_monitored, reward_model, normalize_rewards=NORMALIZE_REWARDS)

    vec_env = DummyVecEnv([make_wrapped])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, gamma=policy.gamma)
    vec_env = VecMonitor(vec_env)
    policy.set_env(vec_env)
    callback = TrueRewardCallback()

    it = 0
    T_cumulative_ppo_steps_in_loop = 0

    while T_cumulative_ppo_steps_in_loop < TOTAL_PPO_TIMESTEPS:
        it += 1
        
        fraction_ppo_training_done = min(1.0, max(0.0, T_cumulative_ppo_steps_in_loop / TOTAL_PPO_TIMESTEPS))
        current_min_gap = INITIAL_MIN_GAP - (INITIAL_MIN_GAP - FINAL_MIN_GAP) * fraction_ppo_training_done
        current_min_gap = max(current_min_gap, FINAL_MIN_GAP)
        
        print(f"Iteration {it}: PPO Timesteps {T_cumulative_ppo_steps_in_loop}/{TOTAL_PPO_TIMESTEPS}, Using MIN_GAP: {current_min_gap:.2f}, Total Prefs: {len(pref_ds)}/{TOTAL_TARGET_PAIRS}")

        policy.learn(
            total_timesteps=PPO_TIMESTEPS_PER_ITER,
            reset_num_timesteps=False,
            callback=callback
        )

        T_cumulative_ppo_steps_in_loop += PPO_TIMESTEPS_PER_ITER

        # collect new clips
        new_full_episodes = collect_clips(policy, NUM_EPISODES_TO_COLLECT_PER_UPDATE, env_id=env_id, max_episode_steps=MAX_EPISODE_STEPS)
        new_segments = extract_segments_from_episodes(
            new_full_episodes,
            SEGMENT_LEN,
            EXTRACT_SEGMENTS,
            TARGET_NUM_SEGMENTS_IF_EXTRACTING_PER_UPDATE if EXTRACT_SEGMENTS else None
        )
        clips_ds.extend(new_segments)

        all_results = vec_env.env_method("get_and_clear_episode_rewards")
        true_rewards = [r for res in all_results for r in res[0]]
        pred_rewards = [r for res in all_results for r in res[1]]
        if true_rewards and pred_rewards:
            plot_true_vs_pred(true_rewards, pred_rewards, results_dir, it, writer=writer)
        else:
            print(f"Iteration {it}: Not enough true/predicted rewards to plot true vs pred.")

        if USE_BINNING:
            bins = create_bins(bins, clips_ds, results_dir, it, NUM_BINS)

        segment_rewards_for_plot = [clip_return(c) for c in new_segments]
        plot_rewards(segment_rewards_for_plot, results_dir, it, writer=writer)

        rate_scaling_factor = REFERENCE_TIMESTEPS_FOR_RATE / (T_cumulative_ppo_steps_in_loop + REFERENCE_TIMESTEPS_FOR_RATE)
        _target_pairs_for_iter = round(BASE_PAIRS_PER_ITERATION_SCALER * rate_scaling_factor)
        _target_pairs_for_iter = max(1, _target_pairs_for_iter) # Ensure we try to collect at least 1 pair

        remaining_needed_overall = TOTAL_TARGET_PAIRS - len(pref_ds)
        target_pairs_this_iter = min(_target_pairs_for_iter, max(0, remaining_needed_overall)) 
        new_prefs = []

        if target_pairs_this_iter > 0 and clips_ds:
            if args.random:
                if USE_BINNING:
                    if not bins:
                        print(f"Warning: Iteration {it}: USE_BINNING is True, but bins are None/empty. Skipping preference generation.")
                    else:
                        if (NUM_BINS * (NUM_BINS-1)) > 0:
                            current_loop_samples_per_bin = math.ceil(target_pairs_this_iter / (NUM_BINS * (NUM_BINS - 1)))
                            current_loop_samples_per_bin = max(1, current_loop_samples_per_bin)
                        else:
                            current_loop_samples_per_bin = target_pairs_this_iter if NUM_BINS == 1 else 0
                        
                        if current_loop_samples_per_bin > 0:
                            new_prefs, _, _ = create_preferences(
                                bins,
                                num_samples_per_other_bin=current_loop_samples_per_bin,
                                min_gap=current_min_gap
                            )
                else:
                    print(it)
                    num_rand_iter_loop = target_pairs_this_iter
                    if T_cumulative_ppo_steps_in_loop < 300_000:
                        new_prefs = sample_random_preferences(clips_ds, num_rand_iter_loop, current_min_gap)
                    else:
                        print(f"Using BALD...")
                        effective_bald_k = min(BALD_K, num_rand_iter_loop)
                        cand_pairs = []
                        if clips_ds:
                            cand_pairs = select_variance_pairs(
                                clips_ds, reward_model,
                                pool_size=num_rand_iter_loop,
                                K=effective_bald_k, T=BALD_T,
                                device=device
                            )
                        if cand_pairs:
                            _annotated_prefs, _, _ = annotate_given_pairs(cand_pairs, min_gap=current_min_gap)
                            new_prefs = _annotated_prefs
                            print(f"Iteration {it}: BALD targeted {target_pairs_this_iter} (effective K {effective_bald_k}), selected {len(new_prefs)} pairs.")
                        

                print(f"Iteration {it}: Targeted {target_pairs_this_iter} random pairs, sampled {len(new_prefs)}.")
            else:
                pool = BALD_POOL_SIZE // 2
                effective_bald_k = min(BALD_K, target_pairs_this_iter) # Cap BALD's K by current target
                
                cand_pairs = []
                if clips_ds:
                    cand_pairs = select_active_pairs(
                        clips_ds, reward_model,
                        pool_size=pool,
                        K=effective_bald_k, T=BALD_T, # Use effective_bald_k
                        device=device
                    )

                if cand_pairs:
                    _annotated_prefs, _, _ = annotate_given_pairs(cand_pairs, min_gap=current_min_gap)
                    new_prefs = _annotated_prefs
                    print(f"Iteration {it}: BALD targeted {target_pairs_this_iter} (effective K {effective_bald_k}), selected {len(new_prefs)} pairs.")
                else:
                    print(f"Iteration {it}: BALD returned no pairs this round (or clips_ds was empty).")

        for c1, c2, p in new_prefs:
            pref_ds.add(c1, c2, p)

        reward_model, reward_logger_iteration = train_reward_model_batched(
            reward_model, pref_ds, device=device,
            epochs=50, patience=7, optimizer=optimizer,
            regularization_weight=REGULARIZATION_WEIGHT,
            logger=policy.logger, iteration=reward_logger_iteration
        )
        for sub in vec_env.envs:
            sub.reward_model = reward_model

        if clips_ds:
            plot_correlation_by_bin(clips_ds, reward_model, it, results_dir, writer=writer)
        else:
            print(f"Iteration {it}: No clips available to generate correlation plot.")

        policy.save(os.path.join(results_dir, f"ppo_{env_id}_iter_{it}.zip"))
        torch.save(
            reward_model.state_dict(),
            os.path.join(results_dir, f"rm_iter_{it}.pth")
        )

    policy.save(os.path.join(results_dir, f"ppo_final_{timestamp}.zip"))
    torch.save(
        reward_model.state_dict(),
        os.path.join(results_dir, f"rm_final_{timestamp}.pth")
    )

    video_folder = os.path.join(results_dir, "final_eval_video")
    os.makedirs(video_folder, exist_ok=True)
    eval_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    eval_env = gym.wrappers.RecordVideo(
        eval_env, video_folder, episode_trigger=lambda e: e == 0,
        name_prefix=f"final-{env_id}-eval"
    )
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = eval_env.step(action)
        done = term or trunc
    eval_env.close()
    print(f"Video saved to {video_folder}/")


if __name__ == "__main__":
    main()

