import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tb_utils import load_tensorboard_scalars, build_bald_time_series_multi

# TensorBoard imports (reuse robust pattern)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator  # type: ignore
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False


def plot_desired_modality_counts_v2(all_desired, output_path, max_bars=None):
    if all_desired is None or all_desired.empty:
        print("No desired modality data to plot for v2.")
        return
    # Aggregate by iteration across runs
    agg = all_desired.groupby('iteration').apply(
        lambda g: pd.Series({
            'pref_mean': g['desired_pref'].mean(),
            'pref_std': g['desired_pref'].std(ddof=1),
            'pref_n': g['desired_pref'].count(),
            'eval_mean': g['desired_eval'].mean(),
            'eval_std': g['desired_eval'].std(ddof=1),
            'eval_n': g['desired_eval'].count()
        })
    ).reset_index()
    for m in ['pref','eval']:
        agg[f'{m}_stderr'] = agg[f'{m}_std'] / np.sqrt(agg[f'{m}_n'].clip(lower=1))
        agg[f'{m}_ci'] = 1.96 * agg[f'{m}_stderr']
    if max_bars is not None and len(agg) > max_bars:
        # Downsample uniformly
        idxs = np.linspace(0, len(agg)-1, max_bars).astype(int)
        agg = agg.iloc[idxs]

    x = agg['iteration']
    pref = agg['pref_mean']
    evalv = agg['eval_mean']

    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(x, pref, color='#E69F00', label='Preference Desired (mean)')
    ax.bar(x, evalv, bottom=pref, color='#4477AA', label='Evaluative Desired (mean)')

    # Plot CI for the preference mean on the dividing line
    ax.errorbar(x, pref, yerr=agg['pref_ci'], fmt='none', ecolor='black', elinewidth=1, capsize=2,
                label='95% CI for Preference Mean')

    ax.set_xlabel('Training Iteration (desired update index)')
    ax.set_ylabel('Desired Feedback Items per Iteration (mean)')
    ax.set_title('Desired Modality per Iteration (Stacked Mean with 95% CI for Preference)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved desired modality counts plot (v2) to {output_path}")
    plt.close(fig)

# ---------------------------------------------
# Data loading utilities are provided by tb_utils.load_tensorboard_scalars
# (Original implementation moved to shared module to avoid duplication.)
# ---------------------------------------------


# ---------------------------------------------
# Percentages of modalities over timesteps
# ---------------------------------------------

def build_percentage_time_series(dfs, pref_tag, eval_tag):
    """Compute percentage of preference/evaluative items over total vs training step.

    Returns long-form DataFrame with columns: modality, step, mean_value, ci_low, ci_high, n
    """
    if pref_tag not in dfs and eval_tag not in dfs:
        print("Neither preference nor evaluative count metrics found.")
        return pd.DataFrame()

    # Build per-run percentage series
    run_ids = set()
    if pref_tag in dfs:
        run_ids.update(dfs[pref_tag]['run_id'].unique())
    if eval_tag in dfs:
        run_ids.update(dfs[eval_tag]['run_id'].unique())

    per_run = []
    for rid in sorted(run_ids):
        pref_df = dfs.get(pref_tag, pd.DataFrame())
        pref_df = pref_df[pref_df['run_id'] == rid][['step', 'value']].rename(columns={'value': 'pref'}) if not pref_df.empty else pd.DataFrame(columns=['step','pref'])
        eval_df = dfs.get(eval_tag, pd.DataFrame())
        eval_df = eval_df[eval_df['run_id'] == rid][['step', 'value']].rename(columns={'value': 'eval'}) if not eval_df.empty else pd.DataFrame(columns=['step','eval'])
        if pref_df.empty and eval_df.empty:
            continue
        merged = pd.merge(pref_df, eval_df, on='step', how='outer').sort_values('step').reset_index(drop=True)
        merged['pref'] = merged['pref'].ffill().fillna(0)
        merged['eval'] = merged['eval'].ffill().fillna(0)
        total = (merged['pref'] + merged['eval']).replace(0, np.nan)
        merged['pref_pct'] = merged['pref'] / total
        merged['eval_pct'] = merged['eval'] / total
        merged['run_id'] = rid
        per_run.append(merged[['run_id','step','pref_pct','eval_pct']])

    if not per_run:
        return pd.DataFrame()
    all_pct = pd.concat(per_run, ignore_index=True)

    # Aggregate across runs by exact step values
    frames = []
    for col, modality in [('pref_pct','preference'), ('eval_pct','evaluative')]:
        g = all_pct.groupby('step')[col].agg(['mean','std','count']).reset_index()
        g.rename(columns={'mean':'mean_value','count':'n'}, inplace=True)
        g['std'] = g['std'].fillna(0.0)
        g['stderr'] = g.apply(lambda r: (r['std']/np.sqrt(r['n'])) if r['n'] > 0 else 0.0, axis=1)
        g['ci'] = 1.96 * g['stderr']
        g['ci_low'] = (g['mean_value'] - g['ci']).clip(lower=0, upper=1)
        g['ci_high'] = (g['mean_value'] + g['ci']).clip(lower=0, upper=1)
        g['modality'] = modality
        frames.append(g[['modality','step','mean_value','ci_low','ci_high','n']])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(['modality','step'])


def plot_percentage_time_series(ts_df, output_path):
    if ts_df.empty:
        print("No percentage time series data to plot.")
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {'preference': '#4C72B0', 'evaluative': '#55A868'}
    for modality, sub in ts_df.groupby('modality'):
        sub = sub.sort_values('step')
        ax.plot(sub['step'], sub['mean_value'] * 100.0, label=f"{modality.capitalize()} %", color=colors.get(modality, None))
        ax.fill_between(sub['step'], sub['ci_low'] * 100.0, sub['ci_high'] * 100.0, color=colors.get(modality, None), alpha=0.25)
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Percentage of Items (%)')
    ax.set_title('Preference vs Evaluative Composition Over Timesteps (Mean ± 95% CI)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved percentage time series plot to {output_path}")
    plt.close(fig)


# ---------------------------------------------
# BALD time series (normalized + unnormalized)
# ---------------------------------------------

# ---------------------------------------------
# BALD time series aggregation moved to tb_utils.build_bald_time_series_multi
# (Original implementation moved to shared module.)
# ---------------------------------------------


def plot_bald_time_series_multi(ts_df, output_path):
    if ts_df.empty:
        print("No BALD time series data to plot.")
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    # A small palette; extend if needed
    palette = [
        '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD'
    ]
    for i, (series, sub) in enumerate(ts_df.groupby('series')):
        sub = sub.sort_values('step')
        color = palette[i % len(palette)]
        ax.plot(sub['step'], sub['mean_value'], label=series, color=color)
        ax.fill_between(sub['step'], sub['ci_low'], sub['ci_high'], color=color, alpha=0.25)
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Average BALD Score')
    ax.set_title('BALD Scores Over Training (Mean ± 95% CI)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved BALD time series plot to {output_path}")
    plt.close(fig)


# ---------------------------------------------
# Main CLI
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Adaptive Ratio: Unified Active Learning Visualization')
    parser.add_argument('log_dir', help='Root folder containing TensorBoard event subdirectories')
    parser.add_argument('--output-dir', default='.', help='Directory to save plots')
    # Count tags for percentages
    parser.add_argument('--pref-count-tag', default='params/num_train_pref_data', help='Preference count metric tag')
    parser.add_argument('--eval-count-tag', default='params/num_train_eval_data', help='Evaluative count metric tag')
    # BALD tags (preference)
    parser.add_argument('--pref-bald-norm-tag', default='active_learning/avg_bald_score_bits', help='Preference BALD normalized tag (bits)')
    parser.add_argument('--pref-bald-unnorm-tag', default='active_learning/avg_bald_score_nats', help='Preference BALD unnormalized tag (nats)')
    parser.add_argument('--pref-bald-fallback', default='active_learning/avg_bald_score', help='Fallback tag if normalized not present')
    # BALD tags (evaluative)
    parser.add_argument('--eval-bald-norm-tag', default='active_learning/avg_evaluative_bald_score', help='Evaluative BALD normalized tag')
    parser.add_argument('--eval-bald-unnorm-tag', default=None, help='Evaluative BALD unnormalized tag (if available)')
    parser.add_argument('--no-percentages', action='store_true', help='Skip percentages plot')
    parser.add_argument('--no-bald', action='store_true', help='Skip BALD time series plot')

    args = parser.parse_args()

    needed_tags = []
    if not args.no_percentages:
        needed_tags += [args.pref_count_tag, args.eval_count_tag]
    if not args.no_bald:
        needed_tags += [
            args.pref_bald_norm_tag,
            args.pref_bald_unnorm_tag,
            args.pref_bald_fallback,
        ]
        if args.eval_bald_norm_tag:
            needed_tags.append(args.eval_bald_norm_tag)
        if args.eval_bald_unnorm_tag:
            needed_tags.append(args.eval_bald_unnorm_tag)
    needed_tags = [t for t in set(needed_tags) if t]

    dfs = load_tensorboard_scalars(args.log_dir, tags=needed_tags)
    if not dfs:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.no_percentages:
        ts_pct = build_percentage_time_series(dfs, args.pref_count_tag, args.eval_count_tag)
        plot_percentage_time_series(ts_pct, os.path.join(args.output_dir, 'modality_percentage_over_timesteps.png'))

    if not args.no_bald:
        # Resolve preference normalized with fallback
        pref_norm_tag = args.pref_bald_norm_tag if args.pref_bald_norm_tag in dfs else (
            args.pref_bald_fallback if args.pref_bald_fallback in dfs else None
        )
        tag_specs = []
        if pref_norm_tag:
            tag_specs.append((pref_norm_tag, 'Preference BALD (normalized)'))
        if args.pref_bald_unnorm_tag and args.pref_bald_unnorm_tag in dfs:
            tag_specs.append((args.pref_bald_unnorm_tag, 'Preference BALD (unnormalized)'))
        # Evaluative
        if args.eval_bald_norm_tag and args.eval_bald_norm_tag in dfs:
            tag_specs.append((args.eval_bald_norm_tag, 'Evaluative BALD (normalized)'))
        if args.eval_bald_unnorm_tag and args.eval_bald_unnorm_tag in dfs:
            tag_specs.append((args.eval_bald_unnorm_tag, 'Evaluative BALD (unnormalized)'))

        ts_bald = build_bald_time_series_multi(dfs, tag_specs)
        plot_bald_time_series_multi(ts_bald, os.path.join(args.output_dir, 'bald_time_series_adaptive_ratio.png'))


if __name__ == '__main__':
    main()
