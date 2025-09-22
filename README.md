
# Active Learning-Based RLHF (MuJoCo Environments)

Reinforcement Learning from Human (Simulated) Feedback with unified active learning over *preference* and *evaluative* signals. The code focuses on efficient allocation of a limited feedback budget using uncertainty-driven acquisition (BALD / variance) and adaptive balancing between feedback modalities.

## Key Ideas
* Unified feedback datasets: combine pairwise preferences and scalar evaluative ratings in a single reward model objective.
* Active acquisition: BALD-style information gain on candidate clip pairs / clips (MC Dropout).
* Adaptive ratio: dynamically adjust proportion of preference vs evaluative queries toward a target fraction.
* Reward model variants: single network, MC Dropout.
* MuJoCo continuous control benchmarks (HalfCheetah-v5, Hopper-v4, Walker2d-v5, etc.).

## Repository Structure (selected)
```
rlhf_mujoco/
	train.py                  # Legacy training (single feedback type)
	train_unified.py          # Unified preference + evaluative training loop
	preferences.py            # Preference dataset & annotation utilities
	evaluative.py             # Evaluative (scalar rating) dataset & annotator
	reward.py / reward_unified.py  # Reward model architectures & training
	bald.py / bald_unified.py # Active selection (BALD / entropy / random baselines)
	custom_env.py             # Learned reward wrapper around Gymnasium env
	plots.py                  # Online plotting helpers (bins, correlations)
	configs/                  # Per-environment hyperparameters & feedback ranges
	plotting_functions/       # Offline analysis & comparison plotting scripts
```

## Installation
It is recommended to use a fresh Conda environment (or virtualenv). MuJoCo rendering on headless servers may require EGL.

```bash
git clone https://github.com/MariaGkoulta/rlhf-repo.git
cd rlhf-repo

conda create -n active-rlhf python=3.10 -y
conda activate active-rlhf

pip install -r requirements.txt

## Config System
Environment- and modality-specific hyperparameters live under `rlhf_mujoco/configs/`:
* `base_config.py`: common defaults (feedback pacing, PPO params, min/max gaps, etc.)
* `cheetah.py`, `hopper.py`, `walker.py`, etc.: override total timesteps, segment lengths, reward ranges, feedback type (`FEEDBACK_TYPE`), and whether to use probabilistic modeling.

Important fields:
* `TOTAL_TARGET_PAIRS`: total feedback budget (unified count across modalities)
* `INITIAL_COLLECTION_FRACTION`: fraction (e.g. 0.1) of budget collected before policy fine-tuning loop
* `SEGMENT_LEN`: segment length for clip slicing
* `EVALUATIVE_RATING_RANGE`, `EVALUATIVE_RATING_BINS`: discretization for evaluative signals
* `PREF_BETA`, `EVAL_BETA`: noise parameters (Bradley–Terry and Gaussian)

Switch environments by changing the import inside the chosen training script (e.g. in `train_unified.py` replace `from configs.walker import *` with the desired config).

## Feedback Modalities
1. Preference (pairwise): Annotator chooses which clip in a pair is better (simulated via true return + noise).
2. Evaluative (scalar rating): Annotator assigns a rating in a numeric range.

## Running Unified Training
Edit the config import at the top of `train_unified.py` if needed, then:
```bash
conda activate active-rlhf
python -m rlhf_mujoco/train_unified \
	--results-dir results/cheetah_run1 \
	--log-dir logs/cheetah_run1 \
	--use-bald \
	--total-target-pairs 750
```

Single-modality training:
```bash
python -m rlhf_mujoco/train --results-dir results/cheetah_pref --feedback-type preference
```

## Plotting & Analysis
Offline scripts in `rlhf_mujoco/plotting_functions/` expect one or more run directories containing TensorBoard logs and saved model artifacts.

Examples:
```bash
# Compare multiple strategies
python rlhf_mujoco/plotting_functions/plot_comparison_diagrams.py \
	comparisons/cheetah/ground-truth comparisons/cheetah/active_preferences \
	--legend-names "ground truth" "active prefs" --analyze-rollout

# Adaptive ratio diagnostics
python rlhf_mujoco/plotting_functions/plot_unified_adaptive_ratio.py --runs logs/cheetah_run1 logs/cheetah_run2
```

## Extending
* Add a new feedback type: create a dataset class (mirroring `EvaluativeDataset`), integrate into unified loss.
* Alternative acquisition: implement a scoring function returning per-pair uncertainty; plug into selection routine.
* Different policy algorithm: swap PPO for SAC by adjusting Stable-Baselines3 import & hyperparams.

## License
See `LICENSE` file for details.

## Acknowledgements
Built on top of Gymnasium, Stable-Baselines3, and MuJoCo.

