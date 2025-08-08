import argparse
import os
import datetime
import random
import gymnasium as gym
import torch
import itertools
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

from preferences import (
    PreferenceDataset, clip_return,
    annotate_pairs
)
from utils import TrueRewardCallback, NoSeedArgumentWrapper
from reward import RewardModel, train_reward_model_batched
from bald import select_active_pairs
from plots import plot_correlation_by_bin, plot_rewards, plot_true_vs_pred
from custom_env import LearnedRewardEnv
from reward_ensemble import RewardEnsemble, select_high_variance_pairs
from evaluative import EvaluativeDataset, annotate_evaluative

from preferences import PreferenceDataset, clip_return, annotate_pairs
from reward import RewardModel, train_reward_model_batched
from bald import select_active_pairs, select_active_clips_for_evaluation
from plots import plot_correlation_by_bin, plot_rewards, plot_true_vs_pred, plot_preference_heatmap, plot_bald_evaluative_selection_distribution
from custom_env import LearnedRewardEnv
from utils import TrueRewardCallback, NoSeedArgumentWrapper

from torch.utils.tensorboard import SummaryWriter
from configs.cheetah import *
import shutil

if FEEDBACK_TYPE == "evaluative":
    NORMALIZE_REWARDS = False  # Evaluative feedback does not require reward normalization
    print("Setting NORMALIZE_REWARDS to False for evaluative feedback.")

def collect_clips(policy, num_episodes_to_collect, env_id="Reacher-v4", n_envs=8, max_episode_steps=50):
    if env_id in UNHEALTHY_TERMINATION_ENVS:
        print(f"Setting unhealthy termination to false for {env_id}.")
        def make_env():
            return gym.make(
                env_id,
                render_mode=None,
                max_episode_steps=max_episode_steps,
                terminate_when_unhealthy=TERMINATE_WHEN_UNHEALTHY
            )
    else:
        def make_env():
            return gym.make(env_id, render_mode=None, max_episode_steps=max_episode_steps)

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
        target_num_segments_if_extracting_multiple: Integer, only used if extract_multiple_segments is True.

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
    print(f"Extracted {len(output_segments)} segments of length {segment_len} from {len(episodes)} episodes.")
    return output_segments

def sample_random_preferences(clips, num_samples, min_gap):
    cand_pairs = []
    while len(cand_pairs) < num_samples:
        c1, c2 = random.sample(clips, 2)
        if abs(sum(c1["rews"]) - sum(c2["rews"])) >= min_gap:
            cand_pairs.append((c1, c2))
    prefs, _, _ = annotate_pairs(cand_pairs, min_gap=min_gap)
    return prefs

def sample_evaluative_data(clips, num_samples):
    """Sample clips for evaluative feedback annotation."""
    num_to_sample = min(len(clips), num_samples)
    if num_to_sample == 0:
        return [], [], []
    selected_clips = random.sample(clips, num_to_sample)
    evaluative_data, _, _ = annotate_evaluative(
        selected_clips, 
        num_bins=EVALUATIVE_RATING_BINS,
        rating_range=EVALUATIVE_RATING_RANGE,
    )
    return evaluative_data

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes
        current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def run_training(
    env_id,
    results_dir,
    run_log_dir,
    use_random_sampling,
    # Hyperparameters from config
    ppo_lr, ppo_n_steps, ppo_batch_size, ppo_ent_coef, ppo_n_epochs,
    initial_policy_ts, total_ppo_timesteps, ppo_timesteps_per_iter,
    rm_lr, rm_weight_decay, rm_reg_weight, rm_epochs, rm_patience,
    rm_dropout_prob,
    use_ensemble, reward_ensembles,
    use_bald, bald_k, bald_t,
    total_target_pairs, initial_collection_fraction,
    num_episodes_to_collect_initial, num_episodes_to_collect_per_update,
    extract_segments, segment_len, initial_segment_len, final_segment_len,
    target_num_segments_if_extracting_initial,
    target_num_segments_if_extracting_per_update,
    initial_min_gap, final_min_gap,
    max_episode_steps, normalize_rewards, terminate_when_unhealthy,
    use_probabilistic_model=False,  # Add new parameter
    use_ground_truth=False
):
    writer = SummaryWriter(log_dir=run_log_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not use_ground_truth:
        if FEEDBACK_TYPE == "preference":
            train_pref_ds = PreferenceDataset(device=device, segment_len=final_segment_len)
            val_pref_ds = PreferenceDataset(device=device, segment_len=final_segment_len)
            train_dataset = train_pref_ds
            val_dataset = val_pref_ds
        elif FEEDBACK_TYPE == "evaluative":
            train_eval_ds = EvaluativeDataset(device=device, segment_len=final_segment_len)
            val_eval_ds = EvaluativeDataset(device=device, segment_len=final_segment_len)
            train_dataset = train_eval_ds
            val_dataset = val_eval_ds
        else:
            raise ValueError(f"Unsupported feedback type: {FEEDBACK_TYPE}")

    raw_env = None
    if env_id in UNHEALTHY_TERMINATION_ENVS:
        print(f"Setting unhealthy termination to false for {env_id}.")
        raw_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps, terminate_when_unhealthy=False)
    else:
        raw_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    wrapped_base_raw_env = NoSeedArgumentWrapper(raw_env)
    time_limited_raw_env = TimeLimit(wrapped_base_raw_env, max_episode_steps=max_episode_steps)
    policy = PPO(
        "MlpPolicy",
        time_limited_raw_env,
        verbose=1,
        n_steps=ppo_n_steps,
        batch_size=ppo_batch_size,
        ent_coef=ppo_ent_coef,
        n_epochs=ppo_n_epochs,
        learning_rate=ppo_lr,
        tensorboard_log=run_log_dir
    )
    policy.learn(total_timesteps=initial_policy_ts)
    time_limited_raw_env.close()

    if not use_ground_truth:
        initial_full_episodes = collect_clips(policy, num_episodes_to_collect_initial, env_id=env_id, max_episode_steps=max_episode_steps)
        clips_ds = []
        if extract_segments:
            clips_ds = extract_segments_from_episodes(
                initial_full_episodes,
                segment_len=segment_len,
                extract_multiple_segments=True,
                target_num_segments_if_multiple=target_num_segments_if_extracting_initial
        )
        else:
            clips_ds = initial_full_episodes
        print(f"Collected {len(initial_full_episodes)} initial full episodes.")
        for i, episode_data in enumerate(initial_full_episodes):
            print(f"  Initial Episode {i+1} length: {len(episode_data['acts'])}")

        initial_target_pairs = int(total_target_pairs * initial_collection_fraction)
        print(f"Targeting {initial_target_pairs} initial {FEEDBACK_TYPE} data points.")

        current_min_gap = initial_min_gap
        print(f"Initial {FEEDBACK_TYPE} generation with MIN_GAP: {current_min_gap}")

        clip_rewards = [clip_return(c) for c in clips_ds]
        plot_rewards(past_clip_rewards=clip_rewards, results_dir=results_dir, it=0, writer=writer)

        if FEEDBACK_TYPE == "preference":
            num_rand_initial = initial_target_pairs
            print(f"Initial collection (random): targeting num_rand_initial = {num_rand_initial}")
            if len(clips_ds) >= 2: 
                    _prefs_list = sample_random_preferences(clips_ds, num_rand_initial, current_min_gap) # Use calculated value
                    prefs = _prefs_list
            else:
                print(f"Initial collection (random): clips_ds has fewer than 2 segments ({len(clips_ds)}). Cannot sample pairs.")

            print(f"Generated {len(prefs)} initial preference pairs.")
            random.shuffle(prefs)
            val_split_idx = int(len(prefs) * 0.2)
            val_prefs = prefs[:val_split_idx]
            train_prefs = prefs[val_split_idx:]

            for c1, c2, p in train_prefs:
                train_pref_ds.add(c1, c2, p)
            for c1, c2, p in val_prefs:
                val_pref_ds.add(c1, c2, p)
            
            print(f"Split initial preferences into {len(train_pref_ds)} training and {len(val_pref_ds)} validation pairs.")
        
        elif FEEDBACK_TYPE == "evaluative":
            evaluative_data = sample_evaluative_data(clips_ds, initial_target_pairs)
            print(f"Generated {len(evaluative_data)} initial evaluative data points.")
            
            random.shuffle(evaluative_data)
            val_split_idx = int(len(evaluative_data) * 0.2)
            val_eval_data = evaluative_data[:val_split_idx]
            train_eval_data = evaluative_data[val_split_idx:]

            for clip, rating in train_eval_data:
                train_eval_ds.add(clip, rating)
            for clip, rating in val_eval_data:
                val_eval_ds.add(clip, rating)
            
            print(f"Split initial evaluative data into {len(train_eval_ds)} training and {len(val_eval_ds)} validation samples.")

    obs_dim = policy.observation_space.shape[0]
    act_dim = policy.action_space.shape[0]
    reward_logger_iteration = 0

    reward_model = None
    reward_ensemble = None
    optimizer = None
    if not use_ground_truth:
        print(f"Training dataset size: {len(train_dataset)}")
            
        if use_random_sampling or use_bald:
            reward_model = RewardModel(obs_dim, act_dim, dropout_prob=rm_dropout_prob, probabilistic=use_probabilistic_model)
            optimizer = torch.optim.Adam(reward_model.parameters(), lr=rm_lr, weight_decay=rm_weight_decay)
            reward_model, reward_logger_iteration = train_reward_model_batched(
                reward_model,
                train_dataset,
                val_dataset,
                device=device,
                epochs=rm_epochs,
                patience=rm_patience,
                optimizer=optimizer,
                regularization_weight=rm_reg_weight,
                logger=policy.logger,
                iteration=reward_logger_iteration,
                feedback_type=FEEDBACK_TYPE,
                rating_scale=EVALUATIVE_RATING_SCALE
            )
        else: 
            reward_ensemble = RewardEnsemble(obs_dim, act_dim, num_models=reward_ensembles, dropout_prob=rm_dropout_prob)
            reward_ensemble.train_ensemble(
                train_dataset,
                val_dataset,
                device=device,
                epochs=rm_epochs,
                patience=rm_patience,
                optimizer_lr=rm_lr,
                optimizer_wd=rm_weight_decay,
                regularization_weight=rm_reg_weight,
                logger=policy.logger,
                iteration=reward_logger_iteration,
                feedback_type=FEEDBACK_TYPE
            )

    def make_wrapped():
        if env_id in UNHEALTHY_TERMINATION_ENVS:
            print(f"Setting unhealthy termination to false for {env_id}.")
            base_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps, terminate_when_unhealthy=False)
        else:
            base_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps)
        wrapped_base_raw_env = NoSeedArgumentWrapper(base_env)
        e_time_limited = TimeLimit(wrapped_base_raw_env, max_episode_steps=max_episode_steps)
        e_monitored = Monitor(e_time_limited)

        if use_ground_truth:
            return e_monitored
        if use_random_sampling or use_bald:
            return LearnedRewardEnv(e_monitored, reward_model, normalize_rewards=normalize_rewards)
        else:
            return LearnedRewardEnv(e_monitored, reward_ensemble, normalize_rewards=normalize_rewards)

    vec_env = DummyVecEnv([make_wrapped])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=policy.gamma)
    vec_env = VecMonitor(vec_env)
    policy.set_env(vec_env)
    
    # Only use TrueRewardCallback when not using ground truth
    callback = None if use_ground_truth else TrueRewardCallback(verbose=1)

    it = 0
    T_cumulative_ppo_steps_in_loop = 0

    while T_cumulative_ppo_steps_in_loop < total_ppo_timesteps:
        it += 1
        
        if not use_ground_truth:
            fraction_ppo_training_done = min(1.0, max(0.0, T_cumulative_ppo_steps_in_loop / total_ppo_timesteps))
            current_min_gap = initial_min_gap - (initial_min_gap - final_min_gap) * fraction_ppo_training_done
            current_min_gap = max(current_min_gap, final_min_gap)
            
            if FEEDBACK_TYPE == "preference":
                print(f"Iteration {it}: PPO Timesteps {T_cumulative_ppo_steps_in_loop}/{total_ppo_timesteps}, Using MIN_GAP: {current_min_gap:.2f}, Total Prefs: {len(train_dataset)+len(val_dataset)}/{total_target_pairs}")
            else:
                print(f"Iteration {it}: PPO Timesteps {T_cumulative_ppo_steps_in_loop}/{total_ppo_timesteps}, Total Evaluative Data: {len(train_dataset)+len(val_dataset)}/{total_target_pairs}")
        
        policy.learn(
            total_timesteps=ppo_timesteps_per_iter,
            reset_num_timesteps=False,
            callback=callback
        )

        T_cumulative_ppo_steps_in_loop += ppo_timesteps_per_iter

        if use_ground_truth:
            # For ground truth, get episode rewards from VecMonitor
            all_episode_rewards = []
            for env_idx in range(vec_env.num_envs):
                if hasattr(vec_env, 'get_episode_rewards') and callable(getattr(vec_env, 'get_episode_rewards')):
                    episode_rewards = vec_env.get_episode_rewards()
                    all_episode_rewards.extend(episode_rewards)
            
            if all_episode_rewards:
                mean_episode_reward = sum(all_episode_rewards) / len(all_episode_rewards)
                policy.logger.record("eval/mean_true_reward", mean_episode_reward)
                print(f"Ground truth mean episode reward: {mean_episode_reward}")
            continue

        current_segment_len = int(initial_segment_len + (final_segment_len - initial_segment_len) * fraction_ppo_training_done)
        policy.logger.record("params/segment_length", current_segment_len)
        print(f"Current segment length: {current_segment_len}")

        # collect new clips
        new_full_episodes = collect_clips(policy, num_episodes_to_collect_per_update, env_id=env_id, max_episode_steps=max_episode_steps)
        new_segments = extract_segments_from_episodes(
            new_full_episodes,
            current_segment_len,
            extract_segments,
            target_num_segments_if_extracting_per_update if extract_segments else None
        )

        import numpy as np
        def stack_obs_acts(clip, device='cpu'):
            obs = torch.tensor(np.stack(clip['obs']), dtype=torch.float32, device=device)
            acts = torch.tensor(np.stack(clip['acts']), dtype=torch.float32, device=device)
            return obs, acts

        if use_random_sampling or use_bald:
            if reward_model is None:
                raise ValueError("Reward model is not initialized. Cannot predict rewards.")
            past_clip_preds = []
            for clip in clips_ds:
                s, a = stack_obs_acts(clip, device)
                if reward_model.probabilistic:
                    r_mean, r_var = reward_model(s, a)
                    r = r_mean.sum().item()
                else:
                    r = reward_model(s, a).sum().item()
                past_clip_preds.append(r)
            new_clip_preds = []
            for segment in new_segments:
                s, a = stack_obs_acts(segment, device)
                if reward_model.probabilistic:
                    r_mean, r_var = reward_model(s, a)
                    r = r_mean.sum().item()
                else:
                    r = reward_model(s, a).sum().item()
                new_clip_preds.append(r)
        else:
            raise ValueError("Reward ensemble is not supported in this context. Please use a reward model for prediction.")

        past_clip_rewards = [clip_return(c) for c in clips_ds]
        new_clip_rewards = [clip_return(c) for c in new_segments]
        plot_rewards(past_clip_rewards=past_clip_rewards, new_clip_rewards=new_clip_rewards, results_dir=results_dir, it=it, writer=writer, reward_type="true")
        plot_rewards(past_clip_rewards=past_clip_preds, new_clip_rewards=new_clip_preds, results_dir=results_dir, it=it, writer=writer, reward_type="predicted")


        plot_true_vs_pred(past_true_rewards=past_clip_rewards, past_pred_rewards=past_clip_preds,
                          new_true_rewards=new_clip_rewards, new_pred_rewards=new_clip_preds,
                          results_dir=results_dir, it=it, writer=writer)

        clips_ds.extend(new_segments)

        all_results = vec_env.env_method("get_and_clear_episode_rewards")
        true_rewards = [r for res in all_results for r in res[0]]
        pred_rewards = [r for res in all_results for r in res[1]]
        if true_rewards and pred_rewards:
            plot_true_vs_pred(past_true_rewards=true_rewards, past_pred_rewards=pred_rewards, results_dir=results_dir, it=it, writer=writer, reward_type="from ppo episode rewards")
        else:
            print(f"Iteration {it}: Not enough true/predicted rewards to plot true vs pred.")


        num_iters_left = (total_ppo_timesteps - T_cumulative_ppo_steps_in_loop) / ppo_timesteps_per_iter
        remaining_needed_overall = total_target_pairs - len(train_dataset)
        if num_iters_left > 0:
            target_points_this_iter = max(0, round(remaining_needed_overall / num_iters_left))
        else:
            target_points_this_iter = 0
        new_prefs = []
        new_evaluative_data = []

        if FEEDBACK_TYPE == "preference":
            if target_points_this_iter > 0 and clips_ds:
                num_rand_iter_loop = target_points_this_iter

                if use_random_sampling:
                    new_prefs = sample_random_preferences(clips_ds, num_rand_iter_loop, current_min_gap)
                    print(f"Iteration {it}: sampled {len(new_prefs)}.")

                if use_bald:
                    print(f"Using BALD for active learning...")
                    effective_bald_k = min(bald_k, num_rand_iter_loop)
                    cand_pairs = []
                    if clips_ds:
                        cand_pairs = select_active_pairs(
                            clips_ds, reward_model,
                            pool_size=num_rand_iter_loop,
                            K=effective_bald_k, T=bald_t,
                            device=device,
                            logger=policy.logger,
                            iteration=reward_logger_iteration,
                            results_dir=results_dir
                        )
                    if cand_pairs:
                        _annotated_prefs, _, rewards_log = annotate_pairs(cand_pairs, min_gap=current_min_gap)
                        new_prefs = _annotated_prefs
                        print(f"Iteration {it}: BALD targeted {target_points_this_iter} (effective K {effective_bald_k}), selected {len(new_prefs)} pairs.")

                        if rewards_log:
                            plot_preference_heatmap(rewards_log, results_dir, it, range_min=-20, range_max=-2)
                    print(f"Iteration {it}: Targeted {target_points_this_iter} bald pairs, sampled {len(new_prefs)}.")
        
                if use_ensemble:
                    print(f"Using ensemble for active learning...")
                    new_prefs = select_high_variance_pairs(
                        clips_ds,
                        reward_ensemble,
                        target_points_this_iter,
                        current_min_gap,
                        logger=policy.logger,
                        iteration=reward_logger_iteration
                    )
                    print(f"Iteration {it}: Selected {len(new_prefs)} high-variance pairs using ensemble.")

            for c1, c2, p in new_prefs:
                if random.random() < 0.8:
                    train_pref_ds.add(c1, c2, p)
                else:
                    val_pref_ds.add(c1, c2, p)

            print(f"Iteration {it}: Added {len(new_prefs)} new preference pairs. Train dataset size: {len(train_pref_ds)}, Val dataset size: {len(val_pref_ds)}")

        elif FEEDBACK_TYPE == "evaluative":
            if target_points_this_iter > 0 and clips_ds:
                if use_bald:
                    print(f"Using BALD for active evaluative feedback selection...")
                    selected_clips, all_rewards, selected_rewards = select_active_clips_for_evaluation(
                        clips_ds,
                        reward_model,
                        K=target_points_this_iter,
                        T=bald_t,
                        device=device,
                        logger=policy.logger,
                        iteration=reward_logger_iteration,
                        gamma=DISCOUNT_FACTOR,
                        rating_range= EVALUATIVE_RATING_RANGE
                    )
                    if all_rewards and selected_rewards:
                        plot_bald_evaluative_selection_distribution(
                            all_rewards, selected_rewards, results_dir, it, writer=writer
                    )
                    new_evaluative_data, _, _ = annotate_evaluative(
                        selected_clips,
                        num_bins=EVALUATIVE_RATING_BINS,
                        rating_range=EVALUATIVE_RATING_RANGE,
                    )
                    print(f"Iteration {it}: BALD selected {len(new_evaluative_data)} clips for evaluation.")
                else:  # Fallback to random sampling for evaluative feedback
                    new_evaluative_data = sample_evaluative_data(clips_ds, target_points_this_iter)
                    print(f"Iteration {it}: Randomly generated {len(new_evaluative_data)} new evaluative data points.")

            for clip, rating in new_evaluative_data:
                if random.random() < 0.8:
                    train_eval_ds.add(clip, rating)
                else:
                    val_eval_ds.add(clip, rating)
                
        policy.logger.record("params/num_train_data", len(train_dataset))

        if use_random_sampling or use_bald:
            reward_model, reward_logger_iteration = train_reward_model_batched(
                reward_model,
                train_dataset,
                val_dataset,
                device=device,
                epochs=50,
                patience=7,
                optimizer=optimizer,
                regularization_weight=rm_reg_weight,
                logger=policy.logger,
                iteration=reward_logger_iteration,
                feedback_type=FEEDBACK_TYPE,
                rating_scale=EVALUATIVE_RATING_SCALE,
            )
            for sub in vec_env.envs:
                sub.reward_model = reward_model
            policy.save(os.path.join(results_dir, f"ppo_{env_id}_iter_{it}.zip"))
            torch.save(reward_model.state_dict(), os.path.join(results_dir, f"rm_iter_{it}.pth"))
            plot_correlation_by_bin(clips_ds, reward_model, it, results_dir, writer=writer)
        else:
            reward_ensemble.train_ensemble(
                train_dataset,
                val_dataset,
                device=device,
                epochs=rm_epochs,
                patience=rm_patience,
                optimizer_lr=rm_lr,
                optimizer_wd=rm_weight_decay,
                regularization_weight=rm_reg_weight,
                logger=policy.logger,
                iteration=reward_logger_iteration,
                feedback_type=FEEDBACK_TYPE
            )
            for sub in vec_env.envs:
                sub.reward_model = reward_ensemble
            plot_correlation_by_bin(clips_ds, reward_ensemble, it, results_dir, writer=writer)
            policy.save(os.path.join(results_dir, f"ppo_{env_id}_iter_{it}.zip"))
            torch.save(reward_ensemble.state_dict(), os.path.join(results_dir, f"rm_iter_{it}.pth"))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    policy.save(os.path.join(results_dir, f"ppo_final_{timestamp}.zip"))
    if not use_ground_truth:
        final_model = reward_ensemble if use_ensemble else reward_model
        torch.save(
            final_model.state_dict(),
            os.path.join(results_dir, f"rm_final_{timestamp}.pth")
        )

    video_folder = os.path.join(results_dir, "final_eval_video")
    os.makedirs(video_folder, exist_ok=True)
    eval_env = None
    if env_id in UNHEALTHY_TERMINATION_ENVS:
        eval_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps, terminate_when_unhealthy=False)
    else:
        eval_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    eval_env = gym.wrappers.RecordVideo(
        eval_env, video_folder, episode_trigger=lambda e: e == 0,
        name_prefix=f"final-{env_id}-eval"
    )
    total_reward = 0
    num_eval_episodes = 10
    for _ in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(action)
            episode_reward += reward
            done = term or trunc
        total_reward += episode_reward
    mean_reward = total_reward / num_eval_episodes
    print(f"Final evaluation mean reward: {mean_reward}")
    eval_env.close()
    print(f"Video saved to {video_folder}/")
    return mean_reward

def main():

    env_id = ENV_ID

    parser = argparse.ArgumentParser(
        description=f"Train PPO on {env_id} with random, BALD-based, or ensemble-based preference sampling"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--random", action="store_true",
                       help="sample preference pairs uniformly at random")
    group.add_argument("--bald", action="store_true",
                       help="sample preference pairs via BALD active learning")
    group.add_argument("--ensemble", action="store_true",
                       help="sample preference pairs via ensemble-based active learning")
    group.add_argument("--ground-truth", action="store_true",
                       help="run with ground truth reward")
    args = parser.parse_args()

    use_ground_truth = False
    if args.bald:
        experiment_type = "active_bald"
        use_bald = True
        use_ensemble = False
        use_random_sampling = False
    elif args.ensemble:
        experiment_type = "ensemble"
        use_bald = False
        use_ensemble = True
        use_random_sampling = False
    elif args.ground_truth:
        experiment_type = "ground_truth"
        use_bald = False
        use_ensemble = False
        use_random_sampling = False
        use_ground_truth = True
    else: # random
        experiment_type = "random"
        use_bald = False
        use_ensemble = False
        use_random_sampling = True
    
    model_type = ''
    if USE_PROBABILISTIC_MODEL:
        model_type = "probabilistic"
    else:
        model_type = "nonprobabilistic"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_time = datetime.datetime.now().strftime("%Y-%m-%d %H")
    results_dir = f"results/{env_id}_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    config_name = env_id.split('-')[0].lower()
    if config_name == "halfcheetah":
        config_name = "cheetah"
    elif config_name == "walker2d":
        config_name = "walker"
    config_filename = f"{config_name}.py"
    config_src = os.path.join(os.path.dirname(__file__), "configs", config_filename)
    config_dst = os.path.join(results_dir, config_filename)
    shutil.copyfile(config_src, config_dst)

    shared_log_dir = f"./logs/{experiment_time}/{env_id}/"
    run_log_dir = f"{shared_log_dir}{experiment_type}_{timestamp}_{model_type}_{FEEDBACK_TYPE}/"

    if use_bald:
        rm_dropout_prob = BALD_REWARD_MODEL_DROPOUT_PROB
    else:
        rm_dropout_prob = REWARD_MODEL_DROPOUT_PROB
    
    
    run_training(
        env_id=env_id,
        results_dir=results_dir,
        run_log_dir=run_log_dir,
        use_random_sampling=use_random_sampling,
        ppo_lr=PPO_LR,
        ppo_n_steps=PPO_N_STEPS,
        ppo_batch_size=PPO_BATCH_SIZE,
        ppo_ent_coef=PPO_ENT_COEF,
        ppo_n_epochs=PPO_N_EPOCHS,
        initial_policy_ts=INITIAL_POLICY_TS,
        total_ppo_timesteps=TOTAL_PPO_TIMESTEPS,
        ppo_timesteps_per_iter=PPO_TIMESTEPS_PER_ITER,
        rm_lr=REWARD_MODEL_LEARNING_RATE,
        rm_weight_decay=REWARD_MODEL_WEIGHT_DECAY,
        rm_reg_weight=REWARD_MODEL_REGULARIZATION_WEIGHT,
        rm_epochs=REWARD_MODEL_EPOCHS,
        rm_patience=REWARD_MODEL_PATIENCE,
        rm_dropout_prob=rm_dropout_prob,
        reward_ensembles=REWARD_ENSEMBLES,
        use_bald=use_bald,
        use_ensemble=use_ensemble,
        bald_k=BALD_K,
        bald_t=BALD_T,
        total_target_pairs=TOTAL_TARGET_PAIRS,
        initial_collection_fraction=INITIAL_COLLECTION_FRACTION,
        num_episodes_to_collect_initial=NUM_EPISODES_TO_COLLECT_INITIAL,
        num_episodes_to_collect_per_update=NUM_EPISODES_TO_COLLECT_PER_UPDATE,
        extract_segments=EXTRACT_SEGMENTS,
        segment_len=SEGMENT_LEN,
        initial_segment_len=INITIAL_SEGMENT_LEN,
        final_segment_len=FINAL_SEGMENT_LEN,
        target_num_segments_if_extracting_initial=TARGET_NUM_SEGMENTS_IF_EXTRACTING_INITIAL,
        target_num_segments_if_extracting_per_update=TARGET_NUM_SEGMENTS_IF_EXTRACTING_PER_UPDATE,
        initial_min_gap=INITIAL_MIN_GAP,
        final_min_gap=FINAL_MIN_GAP,
        max_episode_steps=MAX_EPISODE_STEPS,
        normalize_rewards=NORMALIZE_REWARDS,
        terminate_when_unhealthy=TERMINATE_WHEN_UNHEALTHY,
        use_probabilistic_model=USE_PROBABILISTIC_MODEL,
        use_ground_truth=use_ground_truth
    )


if __name__ == "__main__":
    main()