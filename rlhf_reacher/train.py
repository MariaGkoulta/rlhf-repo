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
from torch.utils.tensorboard import SummaryWriter

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


from preferences import (
    UPPER_BIN, PreferenceDataset, annotate_preferences, clip_return,
    create_bins, create_preferences, annotate_pairs
)
from reward import RewardModel, train_reward_model_batched
from bald import select_active_pairs, select_variance_pairs
from plots import plot_correlation_by_bin, plot_rewards, plot_true_vs_pred, plot_preference_heatmap
from custom_env import LearnedRewardEnv
from utils import TrueRewardCallback, NoSeedArgumentWrapper

from torch.utils.tensorboard import SummaryWriter
from cheetah_config import *
import shutil

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
    prefs, _, _ = annotate_pairs(cand_pairs, min_gap=min_gap)
    return prefs    

def main():

    env_id = ENV_ID

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
    results_dir = f"{env_id}_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in: {results_dir}")

    config_src = os.path.join(os.path.dirname(__file__), "hopper_config.py")
    config_dst = os.path.join(results_dir, "hopper_config.py")
    shutil.copyfile(config_src, config_dst)

    tensorboard_log_dir = f"./logs/{env_id}/"
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pref_ds = PreferenceDataset(device=device, segment_len=FINAL_SEGMENT_LEN)

    raw_env = None
    if env_id in UNHEALTHY_TERMINATION_ENVS:
        print(f"Setting unhealthy termination to false for {env_id}.")
        raw_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS, terminate_when_unhealthy=False)
    else:
        raw_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    wrapped_base_raw_env = NoSeedArgumentWrapper(raw_env)
    time_limited_raw_env = TimeLimit(wrapped_base_raw_env, max_episode_steps=MAX_EPISODE_STEPS)
    policy = PPO(
        "MlpPolicy",
        time_limited_raw_env,
        verbose=1,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        ent_coef=PPO_ENT_COEF,
        n_epochs=PPO_N_EPOCHS,
        learning_rate=PPO_LR,
        tensorboard_log=f"./logs/{env_id}/"
    )
    policy.learn(total_timesteps=INITIAL_POLICY_TS)
    time_limited_raw_env.close()

    initial_full_episodes = collect_clips(policy, NUM_EPISODES_TO_COLLECT_INITIAL, env_id=env_id, max_episode_steps=MAX_EPISODE_STEPS)
    clips_ds = []
    if EXTRACT_SEGMENTS:
        clips_ds = extract_segments_from_episodes(
            initial_full_episodes,
            segment_len=SEGMENT_LEN,
            extract_multiple_segments=True,
            target_num_segments_if_multiple=TARGET_NUM_SEGMENTS_IF_EXTRACTING_INITIAL
    )
    else:
        clips_ds = initial_full_episodes
    print(f"Collected {len(initial_full_episodes)} initial full episodes.")
    for i, episode_data in enumerate(initial_full_episodes):
        print(f"  Initial Episode {i+1} length: {len(episode_data['acts'])}")

    initial_target_pairs = int(TOTAL_TARGET_PAIRS * INITIAL_COLLECTION_FRACTION)
    print(f"Targeting {initial_target_pairs} initial preference pairs.")

    current_min_gap = INITIAL_MIN_GAP
    print(f"Initial preference generation with MIN_GAP: {current_min_gap}")

    clip_rewards = [clip_return(c) for c in clips_ds]
    plot_rewards(clip_rewards, results_dir, it=0, writer=writer)

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

    obs_dim = policy.observation_space.shape[0]
    act_dim = policy.action_space.shape[0]
    reward_logger_iteration = 0

    if args.random:
        reward_model = RewardModel(obs_dim, act_dim, dropout_prob=REWARD_MODEL_DROPOUT_PROB)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=REWARD_MODEL_LEARNING_RATE, weight_decay=REWARD_MODEL_WEIGHT_DECAY)
        reward_model, reward_logger_iteration = train_reward_model_batched(
            reward_model,
            pref_ds,
            device=device,
            epochs=REWARD_MODEL_EPOCHS,
            patience=REWARD_MODEL_PATIENCE,
            optimizer=optimizer,
            regularization_weight=REWARD_MODEL_REGULARIZATION_WEIGHT,
            logger=policy.logger,
            iteration=reward_logger_iteration
        )
    else: 
        reward_ensemble = RewardEnsemble(obs_dim, act_dim, num_models=REWARD_ENSEMBLES, dropout_prob=REWARD_MODEL_DROPOUT_PROB)
        reward_ensemble.train_ensemble(
            pref_ds,
            device=device,
            epochs=REWARD_MODEL_EPOCHS,
            patience=REWARD_MODEL_PATIENCE,
            optimizer_lr=REWARD_MODEL_LEARNING_RATE,
            optimizer_wd=REWARD_MODEL_WEIGHT_DECAY,
            regularization_weight=REWARD_MODEL_REGULARIZATION_WEIGHT,
            logger=policy.logger,
            iteration=reward_logger_iteration
        )

    def make_wrapped():
        if env_id in UNHEALTHY_TERMINATION_ENVS:
            print(f"Setting unhealthy termination to false for {env_id}.")
            base_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS, terminate_when_unhealthy=False)
        else:
            base_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
        wrapped_base_raw_env = NoSeedArgumentWrapper(base_env)
        e_time_limited = TimeLimit(wrapped_base_raw_env, max_episode_steps=MAX_EPISODE_STEPS)
        e_monitored = Monitor(e_time_limited)

        if args.random:
            return LearnedRewardEnv(e_monitored, reward_model, normalize_rewards=NORMALIZE_REWARDS)
        else:
            return LearnedRewardEnv(e_monitored, reward_ensemble, normalize_rewards=NORMALIZE_REWARDS)

    vec_env = DummyVecEnv([make_wrapped])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, gamma=policy.gamma)
    vec_env = VecMonitor(vec_env)
    policy.set_env(vec_env)
    callback = TrueRewardCallback(patience=PPO_TRAINING_PATIENCE, verbose=1)

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

        current_segment_len = int(INITIAL_SEGMENT_LEN + (FINAL_SEGMENT_LEN - INITIAL_SEGMENT_LEN) * fraction_ppo_training_done)
        policy.logger.record("params/segment_length", current_segment_len)
        print(f"Current segment length: {current_segment_len}")

        # collect new clips
        new_full_episodes = collect_clips(policy, NUM_EPISODES_TO_COLLECT_PER_UPDATE, env_id=env_id, max_episode_steps=MAX_EPISODE_STEPS)
        new_segments = extract_segments_from_episodes(
            new_full_episodes,
            current_segment_len,
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

        segment_rewards_for_plot = [clip_return(c) for c in new_segments]
        plot_rewards(segment_rewards_for_plot, results_dir, it, writer=writer)

        rate_scaling_factor = REFERENCE_TIMESTEPS_FOR_RATE / (T_cumulative_ppo_steps_in_loop + REFERENCE_TIMESTEPS_FOR_RATE)
        _target_pairs_for_iter = round(BASE_PAIRS_PER_ITERATION_SCALER * rate_scaling_factor)
        _target_pairs_for_iter = max(1, _target_pairs_for_iter) # Ensure we try to collect at least 1 pair

        remaining_needed_overall = TOTAL_TARGET_PAIRS - len(pref_ds)
        target_pairs_this_iter = min(_target_pairs_for_iter, max(0, remaining_needed_overall)) 
        new_prefs = []

        if target_pairs_this_iter > 0 and clips_ds:
            num_rand_iter_loop = target_pairs_this_iter
            if args.random:
                if T_cumulative_ppo_steps_in_loop < 10_000_000:
                    new_prefs = sample_random_preferences(clips_ds, num_rand_iter_loop, current_min_gap)
                else:
                    new_prefs = select_high_variance_pairs(clips_ds, reward_ensemble, target_pairs_this_iter, current_min_gap)
                    print(f"Iteration {it}: Selected {len(new_prefs)} high-variance pairs using ensemble.")
                    # print(f"Using BALD...")
                    # effective_bald_k = min(BALD_K, num_rand_iter_loop)
                    # cand_pairs = []
                    # if clips_ds:
                    #     cand_pairs = select_active_pairs(
                    #         clips_ds, reward_model,
                    #         pool_size=num_rand_iter_loop,
                    #         K=effective_bald_k, T=BALD_T,
                    #         device=device,
                    #         logger=policy.logger,
                    #         iteration=reward_logger_iteration
                    #     )
                    # if cand_pairs:
                    #     _annotated_prefs, _, rewards_log = annotate_pairs(cand_pairs, min_gap=current_min_gap)
                    #     new_prefs = _annotated_prefs
                    #     print(f"Iteration {it}: BALD targeted {target_pairs_this_iter} (effective K {effective_bald_k}), selected {len(new_prefs)} pairs.")
                    
                    #     if rewards_log:
                    #         plot_preference_heatmap(rewards_log, results_dir, it, range_min=-20, range_max=-2)

                print(f"Iteration {it}: Targeted {target_pairs_this_iter} random pairs, sampled {len(new_prefs)}.")
            else:
                new_prefs = select_high_variance_pairs(clips_ds, reward_ensemble, target_pairs_this_iter, current_min_gap)
                print(f"Iteration {it}: Selected {len(new_prefs)} high-variance pairs using ensemble.")
                # print(f"Using BALD...")
                # effective_bald_k = min(BALD_K, num_rand_iter_loop)
                # cand_pairs = []
                # if clips_ds:
                #     cand_pairs = select_active_pairs(
                #         clips_ds, reward_model,
                #         pool_size=num_rand_iter_loop,
                #         K=effective_bald_k, T=BALD_T,
                #         device=device,
                #         logger=policy.logger,
                #         iteration=reward_logger_iteration
                #     )
                # if cand_pairs:
                #     _annotated_prefs, _, rewards_log = annotate_pairs(cand_pairs, min_gap=current_min_gap)
                #     new_prefs = _annotated_prefs
                #     print(f"Iteration {it}: BALD targeted {target_pairs_this_iter} (effective K {effective_bald_k}), selected {len(new_prefs)} pairs.")
                
                #     if rewards_log:
                #         plot_preference_heatmap(rewards_log, results_dir, it, range_min=-20, range_max=-2)

        for c1, c2, p in new_prefs:
            pref_ds.add(c1, c2, p)

        if args.random:
            reward_model, reward_logger_iteration = train_reward_model_batched(
                reward_model,
                pref_ds,
                device=device,
                epochs=50,
                patience=7,
                optimizer=optimizer,
                regularization_weight=REWARD_MODEL_REGULARIZATION_WEIGHT,
                logger=policy.logger,
                iteration=reward_logger_iteration
            )
            for sub in vec_env.envs:
                sub.reward_model = reward_model
            policy.save(os.path.join(results_dir, f"ppo_{env_id}_iter_{it}.zip"))
            torch.save(reward_model.state_dict(), os.path.join(results_dir, f"rm_iter_{it}.pth"))
            plot_correlation_by_bin(clips_ds, reward_model, it, results_dir, writer=writer)
        else:
            reward_ensemble.train_ensemble(
                pref_ds,
                device=device,
                epochs=REWARD_MODEL_EPOCHS,
                patience=REWARD_MODEL_PATIENCE,
                optimizer_lr=REWARD_MODEL_LEARNING_RATE,
                optimizer_wd=REWARD_MODEL_WEIGHT_DECAY,
                regularization_weight=REWARD_MODEL_REGULARIZATION_WEIGHT,
                logger=policy.logger,
                iteration=reward_logger_iteration
            )
            for sub in vec_env.envs:
                sub.reward_model = reward_ensemble
            plot_correlation_by_bin(clips_ds, reward_ensemble, it, results_dir, writer=writer)
            policy.save(os.path.join(results_dir, f"ppo_{env_id}_iter_{it}.zip"))
            torch.save(reward_ensemble.state_dict(), os.path.join(results_dir, f"rm_iter_{it}.pth"))

    policy.save(os.path.join(results_dir, f"ppo_final_{timestamp}.zip"))
    torch.save(
        reward_ensemble.state_dict(),
        os.path.join(results_dir, f"rm_final_{timestamp}.pth")
    )

    video_folder = os.path.join(results_dir, "final_eval_video")
    os.makedirs(video_folder, exist_ok=True)
    eval_env = None
    if env_id in UNHEALTHY_TERMINATION_ENVS:
        eval_env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS, terminate_when_unhealthy=False)
    else:
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