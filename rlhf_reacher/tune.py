import optuna
import os
import datetime
import shutil

# Import the training function and config variables
from train import run_training
from cheetah_config import *

def objective(trial: optuna.Trial):
    """
    Defines the objective function for Optuna.
    A trial consists of a set of hyperparameters and the resulting mean reward.
    """

    ppo_lr = trial.suggest_float("ppo_lr", 1e-5, 1e-3, log=True)
    ppo_ent_coef = trial.suggest_float("ppo_ent_coef", 0.0, 0.1)
    ppo_n_epochs = trial.suggest_int("ppo_n_epochs", 5, 20)
    
    rm_lr = trial.suggest_float("rm_lr", 1e-5, 1e-2, log=True)
    rm_weight_decay = trial.suggest_float("rm_weight_decay", 1e-5, 1e-2, log=True)
    rm_reg_weight = trial.suggest_float("rm_reg_weight", 1e-6, 1e-4, log=True)
    rm_dropout_prob = trial.suggest_float("rm_dropout_prob", 0.1, 0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_type = "active_bald" if USE_BALD else "random"
    run_name = f"trial_{trial.number}_{experiment_type}_{timestamp}"
    results_dir = f"optuna_results/{run_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    shared_log_dir = f"./optuna_logs/{ENV_ID}/"
    run_log_dir = f"{shared_log_dir}{run_name}/"

    total_ppo_timesteps_tuning = 5e5 
    ppo_timesteps_per_iter_tuning = 20000

    use_random_sampling_for_tuning = not USE_REWARD_ENSEMBLES

    mean_reward = run_training(
        env_id=ENV_ID,
        results_dir=results_dir,
        run_log_dir=run_log_dir,
        use_random_sampling=use_random_sampling_for_tuning,
        # Tuned Hyperparameters
        ppo_lr=ppo_lr,
        ppo_ent_coef=ppo_ent_coef,
        ppo_n_epochs=ppo_n_epochs,
        rm_lr=rm_lr,
        rm_weight_decay=rm_weight_decay,
        rm_reg_weight=rm_reg_weight,
        rm_dropout_prob=rm_dropout_prob,
        # Fixed Hyperparameters from config
        ppo_n_steps=PPO_N_STEPS,
        ppo_batch_size=PPO_BATCH_SIZE,
        initial_policy_ts=INITIAL_POLICY_TS,
        total_ppo_timesteps=total_ppo_timesteps_tuning,
        ppo_timesteps_per_iter=ppo_timesteps_per_iter_tuning,
        ppo_training_patience=PPO_TRAINING_PATIENCE,
        rm_epochs=REWARD_MODEL_EPOCHS,
        rm_patience=REWARD_MODEL_PATIENCE,
        reward_ensembles=REWARD_ENSEMBLES,
        use_bald=USE_BALD,
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
        terminate_when_unhealthy=TERMINATE_WHEN_UNHEALTHY
    )
    shutil.rmtree(results_dir)
    shutil.rmtree(run_log_dir)
    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=f"{ENV_ID}-tuning"
    )
    study.optimize(objective, n_trials=50)
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Mean Reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    df = study.trials_dataframe()
    df.to_csv(f"optuna_results_{ENV_ID}.csv")
    print(f"\nResults saved to optuna_results_{ENV_ID}.csv")