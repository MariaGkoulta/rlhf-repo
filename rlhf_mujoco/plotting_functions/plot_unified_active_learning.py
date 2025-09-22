import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from tb_utils import load_tensorboard_scalars, build_bald_time_series

# TensorBoard imports (same pattern as existing plotting script)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator  # type: ignore
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False

# ---------------------------------------------
# Data loading utilities are now in tb_utils.load_tensorboard_scalars
# ---------------------------------------------

# ---------------------------------------------
# Modality counts over time (stacked bars)
# ---------------------------------------------

def build_modality_counts_dataframe(dfs, pref_tag, eval_tag):
    if pref_tag not in dfs and eval_tag not in dfs:
        print("Neither preference nor evaluative count metrics found.")
        return pd.DataFrame()

    # Combine counts per run. Use iteration index (order of appearance) instead of raw step for alignment.
    parts = []
    run_ids = set()
    if pref_tag in dfs:
        run_ids.update(dfs[pref_tag]['run_id'].unique())
    if eval_tag in dfs:
        run_ids.update(dfs[eval_tag]['run_id'].unique())

    for rid in sorted(run_ids):
        pref_df = dfs.get(pref_tag, pd.DataFrame())[lambda d: d['run_id'] == rid] if pref_tag in dfs else pd.DataFrame()
        eval_df = dfs.get(eval_tag, pd.DataFrame())[lambda d: d['run_id'] == rid] if eval_tag in dfs else pd.DataFrame()
        # Merge on step (outer) then forward fill so counts persist until updated.
        merged = pd.merge(pref_df[['step','value']].rename(columns={'value':'pref_count'}),
                          eval_df[['step','value']].rename(columns={'value':'eval_count'}),
                          on='step', how='outer') if not pref_df.empty or not eval_df.empty else pd.DataFrame()
        if merged.empty:
            continue
        merged = merged.sort_values('step').reset_index(drop=True)
        merged['pref_count'] = merged['pref_count'].ffill().fillna(0)
        merged['eval_count'] = merged['eval_count'].ffill().fillna(0)
        merged['iteration'] = np.arange(len(merged))
        merged['run_id'] = rid
        parts.append(merged[['run_id','iteration','step','pref_count','eval_count']])
    if not parts:
        return pd.DataFrame()
    all_counts = pd.concat(parts, ignore_index=True)
    return all_counts

def plot_modality_counts(all_counts, output_path, max_bars=None):
    if all_counts.empty:
        print("No modality count data to plot.")
        return
    # Aggregate by iteration across runs (only keep iterations present in at least one run)
    agg = all_counts.groupby('iteration').apply(
        lambda g: pd.Series({
            'pref_mean': g['pref_count'].mean(),
            'pref_std': g['pref_count'].std(ddof=1),
            'pref_n': g['pref_count'].count(),
            'eval_mean': g['eval_count'].mean(),
            'eval_std': g['eval_count'].std(ddof=1),
            'eval_n': g['eval_count'].count()
        })
    ).reset_index()
    # Standard error and 95% CI
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
    total = pref + evalv

    fig, ax = plt.subplots(figsize=(14,6))
    bar_pref = ax.bar(x, pref, color='#E69F00', label='Preference Count (mean)')
    bar_eval = ax.bar(x, evalv, bottom=pref, color='#4477AA', label='Evaluative Count (mean)')

    # Add error bars for each segment: draw as line segments at segment centers
    ax.errorbar(x, pref/2, yerr=agg['pref_ci'], fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax.errorbar(x, pref + evalv/2, yerr=agg['eval_ci'], fmt='none', ecolor='black', elinewidth=1, capsize=2)

    ax.set_xlabel('Training Iteration (count update index)')
    ax.set_ylabel('Cumulative Training Items (mean)')
    ax.set_title('Modality Composition Over Time (Stacked Mean Counts with 95% CI)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved modality counts plot to {output_path}")
    plt.close(fig)

# ---------------------------------------------
# Desired modality counts per iteration (stacked bars, v2 with shared CI)
# ---------------------------------------------

def plot_desired_modality_counts_v2(all_desired, output_path, max_bars=None):
    if all_desired is None or all_desired.empty:
        print("No desired modality data to plot for v2.")
        return
    # Compute per-run preference share per iteration, then aggregate mean ± 95% CI across runs
    df = all_desired.copy()
    df['total'] = df['desired_pref'].fillna(0) + df['desired_eval'].fillna(0)
    df = df[df['total'] > 0]  # drop rows where no desired items were logged
    if df.empty:
        print("No valid (nonzero) desired modality rows to plot for v2.")
        return
    df['pref_share'] = df['desired_pref'] / df['total']

    agg = df.groupby('iteration')['pref_share'].agg(['mean', 'std', 'count']).reset_index()
    agg.rename(columns={'mean': 'pref_prop_mean', 'std': 'pref_prop_std', 'count': 'pref_prop_n'}, inplace=True)
    agg['pref_prop_std'] = agg['pref_prop_std'].fillna(0.0)
    agg['pref_prop_stderr'] = agg.apply(lambda r: (r['pref_prop_std'] / np.sqrt(r['pref_prop_n'])) if r['pref_prop_n'] > 0 else 0.0, axis=1)
    agg['pref_prop_ci'] = 1.96 * agg['pref_prop_stderr']
    if max_bars is not None and len(agg) > max_bars:
        # Downsample uniformly
        idxs = np.linspace(0, len(agg)-1, max_bars).astype(int)
        agg = agg.iloc[idxs]

    x = agg['iteration']
    pref = agg['pref_prop_mean']
    evalv = 1.0 - pref

    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(x, pref, color='#E69F00', label='Preference Share (mean)')
    ax.bar(x, evalv, bottom=pref, color='#4477AA', label='Evaluative Share (mean)')

    # Plot CI for the preference mean on the dividing line
    ax.errorbar(x, pref, yerr=agg['pref_prop_ci'], fmt='none', ecolor='black', elinewidth=1, capsize=2,
                label='95% CI for Preference Share')

    ax.set_xlabel('Training Iteration (desired update index)')
    ax.set_ylabel('Feedback Share per Iteration (mean)')
    ax.set_title('Feedback Type Percentage per Iteration (Mean ± 95% CI on Share)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved desired modality counts plot (v2) to {output_path}")
    plt.close(fig)

# ---------------------------------------------
# BALD score distribution over training stages
# ---------------------------------------------

def build_bald_dataframe(dfs, pref_bald_tag, eval_bald_tag):
    frames = []
    if pref_bald_tag in dfs:
        d = dfs[pref_bald_tag].copy()
        d['modality'] = 'preference'
        frames.append(d)
    if eval_bald_tag in dfs:
        d = dfs[eval_bald_tag].copy()
        d['modality'] = 'evaluative'
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

def select_training_stages(df, num_stages):
    # Determine global step range and select evenly spaced target steps
    steps = df['step'].sort_values().unique()
    if len(steps) == 0:
        return []
    chosen = np.linspace(steps.min(), steps.max(), num_stages).astype(int)
    return chosen

def extract_stage_samples(df, target_steps):
    # For each run & modality, pick the row nearest to each target step
    samples = []
    for modality in df['modality'].unique():
        sub = df[df['modality'] == modality]
        for rid in sub['run_id'].unique():
            run_df = sub[sub['run_id'] == rid]
            for ts in target_steps:
                # nearest step
                idx = (run_df['step'] - ts).abs().idxmin()
                row = run_df.loc[idx]
                samples.append({
                    'target_step': ts,
                    'actual_step': row['step'],
                    'value': row['value'],
                    'modality': modality,
                    'run_id': rid
                })
    return pd.DataFrame(samples)

def plot_bald_distribution(bald_df, output_path, num_stages):
    if bald_df.empty:
        print("No BALD score data to plot.")
        return
    stages = select_training_stages(bald_df, num_stages)
    stage_samples = extract_stage_samples(bald_df, stages)
    if stage_samples.empty:
        print("No samples extracted for stages.")
        return

    # Use seaborn boxplot + swarm for distribution across runs per modality per stage
    fig, ax = plt.subplots(figsize=(14,6))
    stage_samples['Stage'] = pd.Categorical(stage_samples['target_step'], ordered=True)

    sns.boxplot(data=stage_samples, x='Stage', y='value', hue='modality', ax=ax)
    sns.stripplot(data=stage_samples, x='Stage', y='value', hue='modality', dodge=True,
                  marker='o', alpha=0.5, linewidth=0, ax=ax)

    # Adjust legend (duplicate due to stripplot)
    handles, labels = ax.get_legend_handles_labels()
    unique = []
    seen = set()
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h,l))
            seen.add(l)
    ax.legend(*zip(*unique), title='Modality')

    ax.set_xlabel('Training Stage (target step)')
    ax.set_ylabel('Average BALD Score')
    ax.set_title('BALD Score Distribution Across Training Stages')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved BALD score distribution plot to {output_path}")
    plt.close(fig)

# ---------------------------------------------
# BALD time series with confidence intervals
# ---------------------------------------------

# ---------------------------------------------
# BALD time series aggregation moved to tb_utils.build_bald_time_series
# ---------------------------------------------

def plot_bald_time_series(ts_df, output_path):
    if ts_df.empty:
        print("No BALD time series data to plot.")
        return
    fig, ax = plt.subplots(figsize=(14,6))
    colors = { 'preference':'#4C72B0', 'evaluative':'#55A868' }
    for modality, sub in ts_df.groupby('modality'):
        sub = sub.sort_values('step')
        ax.plot(sub['step'], sub['mean_value'], label=f"{modality.capitalize()} BALD", color=colors.get(modality, None))
        ax.fill_between(sub['step'], sub['ci_low'], sub['ci_high'], color=colors.get(modality, None), alpha=0.25)
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
# New: Multi-folder BALD plots and queries alignment (Diagrams 3-6)
# ---------------------------------------------

def _aggregate_with_ci_local(df, x_col, y_col='value'):
    if df is None or df.empty:
        return pd.DataFrame(columns=[x_col, 'mean_value', 'ci_low', 'ci_high', 'n'])
    grouped = df.groupby(x_col)[y_col].agg(['mean', 'std', 'count']).reset_index()
    grouped.rename(columns={'mean': 'mean_value', 'count': 'n'}, inplace=True)
    grouped['std'] = grouped['std'].fillna(0.0)
    grouped['stderr'] = grouped.apply(lambda r: (r['std'] / np.sqrt(r['n'])) if r['n'] > 0 else 0.0, axis=1)
    grouped['ci'] = 1.96 * grouped['stderr']
    grouped['ci_low'] = grouped['mean_value'] - grouped['ci']
    grouped['ci_high'] = grouped['mean_value'] + grouped['ci']
    return grouped[[x_col, 'mean_value', 'ci_low', 'ci_high', 'n']]


def _build_combined_bald_by_step_per_dir(dfs, pref_tag=None, eval_tag=None):
    """For a single directory's dfs, build per-run combined BALD series by step.

    If both pref and eval tags exist, take the row-wise mean where available; if only one exists, use it.
    Returns DataFrame with columns [run_id, step, value].
    """
    frames = []
    pref_df = dfs.get(pref_tag) if pref_tag else None
    eval_df = dfs.get(eval_tag) if eval_tag else None
    # Gather run_ids present in either df
    run_ids = set()
    if pref_df is not None:
        run_ids.update(pref_df['run_id'].unique())
    if eval_df is not None:
        run_ids.update(eval_df['run_id'].unique())
    for rid in sorted(run_ids):
        p = pref_df[pref_df['run_id'] == rid][['step', 'value']].rename(columns={'value': 'pref'}) if pref_df is not None else pd.DataFrame(columns=['step', 'pref'])
        e = eval_df[eval_df['run_id'] == rid][['step', 'value']].rename(columns={'value': 'eval'}) if eval_df is not None else pd.DataFrame(columns=['step', 'eval'])
        if p.empty and e.empty:
            continue
        merged = pd.merge(p, e, on='step', how='outer').sort_values('step').reset_index(drop=True)
        # Row-wise mean ignoring NaNs
        merged['value'] = merged[['pref', 'eval']].mean(axis=1, skipna=True)
        merged['run_id'] = rid
        frames.append(merged[['run_id', 'step', 'value']])
    if not frames:
        return pd.DataFrame(columns=['run_id', 'step', 'value'])
    return pd.concat(frames, ignore_index=True)


def _get_total_queries_df(dfs, total_tag='params/num_train_data', pref_count_tag='params/num_train_pref_data', eval_count_tag='params/num_train_eval_data'):
    """Return per-run total queries timeline as DataFrame [run_id, step, total_queries].

    If total_tag exists, use it; otherwise, sum pref and eval counts, forward-filling between updates.
    """
    if total_tag in dfs:
        d = dfs[total_tag][['run_id', 'step', 'value']].copy()
        d.rename(columns={'value': 'total_queries'}, inplace=True)
        return d.sort_values(['run_id', 'step']).reset_index(drop=True)

    pref_df = dfs.get(pref_count_tag)
    eval_df = dfs.get(eval_count_tag)
    if pref_df is None and eval_df is None:
        return pd.DataFrame(columns=['run_id', 'step', 'total_queries'])

    run_ids = set()
    if pref_df is not None:
        run_ids.update(pref_df['run_id'].unique())
    if eval_df is not None:
        run_ids.update(eval_df['run_id'].unique())
    frames = []
    for rid in sorted(run_ids):
        p = pref_df[pref_df['run_id'] == rid][['step', 'value']].rename(columns={'value': 'pref'}) if pref_df is not None else pd.DataFrame(columns=['step', 'pref'])
        e = eval_df[eval_df['run_id'] == rid][['step', 'value']].rename(columns={'value': 'eval'}) if eval_df is not None else pd.DataFrame(columns=['step', 'eval'])
        merged = pd.merge(p, e, on='step', how='outer').sort_values('step').reset_index(drop=True)
        if merged.empty:
            continue
        merged['pref'] = merged['pref'].ffill().fillna(0)
        merged['eval'] = merged['eval'].ffill().fillna(0)
        merged['total_queries'] = merged['pref'] + merged['eval']
        merged['run_id'] = rid
        frames.append(merged[['run_id', 'step', 'total_queries']])
    if not frames:
        return pd.DataFrame(columns=['run_id', 'step', 'total_queries'])
    return pd.concat(frames, ignore_index=True)


def _build_combined_bald_by_queries_per_dir(dfs, pref_tag, eval_tag, total_tag, pref_count_tag, eval_count_tag):
    """For a single directory's dfs, build combined BALD aligned to total queries.

    Returns DataFrame [queries, mean_value, ci_low, ci_high, n].
    """
    bald_per_run = _build_combined_bald_by_step_per_dir(dfs, pref_tag=pref_tag, eval_tag=eval_tag)
    if bald_per_run.empty:
        return pd.DataFrame(columns=['queries', 'mean_value', 'ci_low', 'ci_high', 'n'])
    queries_df = _get_total_queries_df(dfs, total_tag=total_tag, pref_count_tag=pref_count_tag, eval_count_tag=eval_count_tag)
    if queries_df.empty:
        return pd.DataFrame(columns=['queries', 'mean_value', 'ci_low', 'ci_high', 'n'])

    aligned_frames = []
    for rid in sorted(bald_per_run['run_id'].unique()):
        b = bald_per_run[bald_per_run['run_id'] == rid][['step', 'value']].sort_values('step').reset_index(drop=True)
        q = queries_df[queries_df['run_id'] == rid][['step', 'total_queries']].sort_values('step').reset_index(drop=True)
        if b.empty or q.empty:
            continue
        aligned = pd.merge_asof(b, q, on='step', direction='backward')
        aligned = aligned.dropna(subset=['total_queries'])
        if aligned.empty:
            continue
        # Average duplicates per same queries value for this run
        per_q = aligned.groupby('total_queries', as_index=False)['value'].mean()
        per_q.rename(columns={'total_queries': 'queries'}, inplace=True)
        aligned_frames.append(per_q[['queries', 'value']])

    if not aligned_frames:
        return pd.DataFrame(columns=['queries', 'mean_value', 'ci_low', 'ci_high', 'n'])
    all_points = pd.concat(aligned_frames, ignore_index=True)
    return _aggregate_with_ci_local(all_points, x_col='queries', y_col='value')


def _plot_multi_series(df_by_series, x_col, ylabel, title, output_path, colors=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    default_palette = [
        '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#1f77b4', '#2ca02c'
    ]
    for i, (label, agg_df) in enumerate(df_by_series.items()):
        if agg_df is None or agg_df.empty:
            continue
        color = None
        if colors and i < len(colors):
            color = colors[i]
        else:
            color = default_palette[i % len(default_palette)]
        ax.plot(agg_df[x_col], agg_df['mean_value'], label=label, color=color, linewidth=2)
        ax.fill_between(agg_df[x_col], agg_df['ci_low'], agg_df['ci_high'], color=color, alpha=0.2)
    ax.set_xlabel('Number of Queries' if x_col == 'queries' else 'Training Timesteps')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

# New: helpers to plot pref vs eval in same plot per folder (by steps)
def _build_bald_modality_by_step(dfs, tag):
    if not tag or tag not in dfs:
        return pd.DataFrame(columns=['step', 'mean_value', 'ci_low', 'ci_high', 'n'])
    d = dfs[tag][['step', 'value']].copy()
    return _aggregate_with_ci_local(d, x_col='step', y_col='value')


def _plot_pref_eval_by_step(per_dir_dfs, dir_labels, colors, pref_tag, eval_tag, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    # Aggregate across all provided folders/runs per modality
    all_pref = []
    all_eval = []
    for dfs in per_dir_dfs:
        if pref_tag in dfs:
            all_pref.append(dfs[pref_tag][['step', 'value']])
        if eval_tag in dfs:
            all_eval.append(dfs[eval_tag][['step', 'value']])
    pref_agg = pd.DataFrame()
    eval_agg = pd.DataFrame()
    if all_pref:
        pref_cat = pd.concat(all_pref, ignore_index=True)
        pref_agg = _aggregate_with_ci_local(pref_cat, x_col='step', y_col='value')
    if all_eval:
        eval_cat = pd.concat(all_eval, ignore_index=True)
        eval_agg = _aggregate_with_ci_local(eval_cat, x_col='step', y_col='value')

    # Fixed colors per modality
    pref_color = '#E69F00'  # preferences
    eval_color = '#4477AA'  # evaluations

    if not pref_agg.empty:
        ax.plot(pref_agg['step'], pref_agg['mean_value'], color=pref_color, linewidth=2)
        ax.fill_between(pref_agg['step'], pref_agg['ci_low'], pref_agg['ci_high'], color=pref_color, alpha=0.20)
    if not eval_agg.empty:
        ax.plot(eval_agg['step'], eval_agg['mean_value'], color=eval_color, linewidth=2)
        ax.fill_between(eval_agg['step'], eval_agg['ci_low'], eval_agg['ci_high'], color=eval_color, alpha=0.20)

    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    # Legend showing only modality
    modality_handles = [
        Line2D([0], [0], color='#E69F00', linestyle='-', linewidth=2, label='Preference'),
        Line2D([0], [0], color='#4477AA', linestyle='-', linewidth=2, label='Evaluative')
    ]
    ax.legend(handles=modality_handles, title='Modality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

# Stage-based aggregation like plot_bald_distribution, but returns mean±CI series for line plots
def _aggregate_series_by_stages(all_df, num_stages):
    if all_df is None or all_df.empty:
        return pd.DataFrame(columns=['target_step', 'mean_value', 'ci_low', 'ci_high', 'n'])
    steps = all_df['step'].values
    if len(steps) == 0:
        return pd.DataFrame(columns=['target_step', 'mean_value', 'ci_low', 'ci_high', 'n'])
    targets = np.linspace(steps.min(), steps.max(), num_stages).astype(int)
    samples = []
    for rid in all_df['run_id'].unique():
        run_df = all_df[all_df['run_id'] == rid]
        if run_df.empty:
            continue
        for ts in targets:
            idx = (run_df['step'] - ts).abs().idxmin()
            row = run_df.loc[idx]
            samples.append({'target_step': ts, 'value': row['value'], 'run_id': rid})
    if not samples:
        return pd.DataFrame(columns=['target_step', 'mean_value', 'ci_low', 'ci_high', 'n'])
    smp = pd.DataFrame(samples)
    grouped = smp.groupby('target_step')['value'].agg(['mean', 'std', 'count']).reset_index()
    grouped.rename(columns={'mean':'mean_value', 'count':'n'}, inplace=True)
    grouped['std'] = grouped['std'].fillna(0.0)
    grouped['stderr'] = grouped.apply(lambda r: (r['std']/np.sqrt(r['n'])) if r['n']>0 else 0.0, axis=1)
    grouped['ci'] = 1.96 * grouped['stderr']
    grouped['ci_low'] = grouped['mean_value'] - grouped['ci']
    grouped['ci_high'] = grouped['mean_value'] + grouped['ci']
    return grouped[['target_step', 'mean_value', 'ci_low', 'ci_high', 'n']]


def _plot_pref_eval_by_stages(per_dir_dfs, pref_tag, eval_tag, num_stages, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    # Collect all runs for each modality across provided folders
    all_pref = []
    all_eval = []
    for dfs in per_dir_dfs:
        if pref_tag in dfs:
            all_pref.append(dfs[pref_tag][['run_id','step','value']])
        if eval_tag in dfs:
            all_eval.append(dfs[eval_tag][['run_id','step','value']])
    pref_agg = pd.DataFrame()
    eval_agg = pd.DataFrame()
    if all_pref:
        pref_cat = pd.concat(all_pref, ignore_index=True)
        pref_agg = _aggregate_series_by_stages(pref_cat, num_stages)
    if all_eval:
        eval_cat = pd.concat(all_eval, ignore_index=True)
        eval_agg = _aggregate_series_by_stages(eval_cat, num_stages)

    pref_color = '#E69F00'
    eval_color = '#4477AA'
    if not pref_agg.empty:
        ax.plot(pref_agg['target_step'], pref_agg['mean_value'], color=pref_color, linewidth=2)
        ax.fill_between(pref_agg['target_step'], pref_agg['ci_low'], pref_agg['ci_high'], color=pref_color, alpha=0.20)
    if not eval_agg.empty:
        ax.plot(eval_agg['target_step'], eval_agg['mean_value'], color=eval_color, linewidth=2)
        ax.fill_between(eval_agg['target_step'], eval_agg['ci_low'], eval_agg['ci_high'], color=eval_color, alpha=0.20)

    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    modality_handles = [
        Line2D([0], [0], color=pref_color, linestyle='-', linewidth=2, label='Preference'),
        Line2D([0], [0], color=eval_color, linestyle='-', linewidth=2, label='Evaluative')
    ]
    ax.legend(handles=modality_handles, title='Modality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

# ---------------------------------------------
# Desired modality counts per iteration (stacked bars)
# ---------------------------------------------

def build_desired_modality_dataframe(dfs, desired_pref_tag, desired_eval_tag):
    """Build per-run desired counts per modality with an iteration index (update order).

    Returns DataFrame columns: [run_id, iteration, step, desired_pref, desired_eval]
    """
    if desired_pref_tag not in dfs and desired_eval_tag not in dfs:
        print("Neither desired preference nor desired evaluative metrics found.")
        return pd.DataFrame()

    parts = []
    run_ids = set()
    if desired_pref_tag in dfs:
        run_ids.update(dfs[desired_pref_tag]['run_id'].unique())
    if desired_eval_tag in dfs:
        run_ids.update(dfs[desired_eval_tag]['run_id'].unique())

    for rid in sorted(run_ids):
        dp = dfs.get(desired_pref_tag, pd.DataFrame())
        de = dfs.get(desired_eval_tag, pd.DataFrame())
        dp = dp[dp['run_id'] == rid][['step','value']].rename(columns={'value':'desired_pref'}) if not dp.empty else pd.DataFrame(columns=['step','desired_pref'])
        de = de[de['run_id'] == rid][['step','value']].rename(columns={'value':'desired_eval'}) if not de.empty else pd.DataFrame(columns=['step','desired_eval'])
        merged = pd.merge(dp, de, on='step', how='outer')
        if merged.empty:
            continue
        merged = merged.sort_values('step').reset_index(drop=True)
        # Fill NaNs with 0 in case only one modality was logged at an update step
        merged['desired_pref'] = merged['desired_pref'].fillna(0)
        merged['desired_eval'] = merged['desired_eval'].fillna(0)
        merged['iteration'] = np.arange(len(merged))
        merged['run_id'] = rid
        parts.append(merged[['run_id','iteration','step','desired_pref','desired_eval']])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def plot_desired_modality_counts(all_desired, output_path, max_bars=None):
    if all_desired is None or all_desired.empty:
        print("No desired modality data to plot.")
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
        idxs = np.linspace(0, len(agg)-1, max_bars).astype(int)
        agg = agg.iloc[idxs]

    x = agg['iteration']
    pref = agg['pref_mean']
    evalv = agg['eval_mean']

    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(x, pref, color='#E69F00', label='Preference Desired (mean)')
    ax.bar(x, evalv, bottom=pref, color='#4477AA', label='Evaluative Desired (mean)')

    ax.errorbar(x, pref/2, yerr=agg['pref_ci'], fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax.errorbar(x, pref + evalv/2, yerr=agg['eval_ci'], fmt='none', ecolor='black', elinewidth=1, capsize=2)

    ax.set_xlabel('Training Iteration (desired update index)')
    ax.set_ylabel('Desired Feedback Items per Iteration (mean)')
    ax.set_title('Desired Modality per Iteration (Stacked Mean with 95% CI)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved desired modality counts plot to {output_path}")
    plt.close(fig)

# ---------------------------------------------
# Iteration-indexed BALD aggregation (map steps -> iterations)
# ---------------------------------------------

def _derive_iteration_mapping_from_queries(queries_df):
    """Given DataFrame [run_id, step, total_queries], return mapping
    [run_id, step_update, iter_idx] where iter_idx increments whenever total_queries increases.
    """
    if queries_df is None or queries_df.empty:
        return pd.DataFrame(columns=['run_id', 'step_update', 'iter_idx'])
    frames = []
    for rid in sorted(queries_df['run_id'].unique()):
        q = queries_df[queries_df['run_id'] == rid][['step', 'total_queries']].sort_values('step').reset_index(drop=True)
        if q.empty:
            continue
        # Treat the first row as a change; then increment when total_queries changes
        tq = q['total_queries'].fillna(method='ffill').fillna(0)
        changed = tq.ne(tq.shift()).fillna(True)
        updates = q[changed]
        if updates.empty:
            continue
        upd = updates[['step']].copy()
        upd['iter_idx'] = np.arange(len(upd))
        upd['run_id'] = rid
        upd.rename(columns={'step': 'step_update'}, inplace=True)
        frames.append(upd[['run_id', 'step_update', 'iter_idx']])
    if not frames:
        return pd.DataFrame(columns=['run_id', 'step_update', 'iter_idx'])
    return pd.concat(frames, ignore_index=True)


def _aggregate_bald_by_iterations(per_dir_dfs, tag, total_tag='params/num_train_data', pref_count_tag='params/num_train_pref_data', eval_count_tag='params/num_train_eval_data'):
    """Aggregate a single BALD tag across runs using iteration index derived from query count updates.

    Returns DataFrame [iteration, mean_value, ci_low, ci_high, n]
    """
    # Collect metric across dirs
    metric_frames = []
    queries_frames = []
    for dfs in per_dir_dfs:
        if tag in dfs:
            metric_frames.append(dfs[tag][['run_id','step','value']])
        # Build queries per-run using existing helper
        # Reuse _get_total_queries_df to support fallback when total_tag missing
        q = _get_total_queries_df(dfs, total_tag=total_tag, pref_count_tag=pref_count_tag, eval_count_tag=eval_count_tag)
        if not q.empty:
            queries_frames.append(q[['run_id','step','total_queries']])
    if not metric_frames or not queries_frames:
        return pd.DataFrame(columns=['iteration','mean_value','ci_low','ci_high','n'])

    metrics_all = pd.concat(metric_frames, ignore_index=True)
    queries_all = pd.concat(queries_frames, ignore_index=True)
    # Derive iteration mapping per run
    iter_map = _derive_iteration_mapping_from_queries(queries_all)
    if iter_map.empty:
        return pd.DataFrame(columns=['iteration','mean_value','ci_low','ci_high','n'])

    # Map each metric point to the latest (<= step) update step for its run
    frames = []
    for rid in sorted(metrics_all['run_id'].unique()):
        m = metrics_all[metrics_all['run_id'] == rid][['step','value']].sort_values('step').reset_index(drop=True)
        im = iter_map[iter_map['run_id'] == rid][['step_update','iter_idx']].sort_values('step_update').reset_index(drop=True)
        if m.empty or im.empty:
            continue
        aligned = pd.merge_asof(m, im.rename(columns={'step_update':'step'}), on='step', direction='backward')
        aligned = aligned.dropna(subset=['iter_idx'])
        if aligned.empty:
            continue
        # Average duplicates within the same iteration for this run
        per_iter = aligned.groupby('iter_idx', as_index=False)['value'].mean()
        per_iter.rename(columns={'iter_idx': 'iteration'}, inplace=True)
        frames.append(per_iter[['iteration','value']])
    if not frames:
        return pd.DataFrame(columns=['iteration','mean_value','ci_low','ci_high','n'])

    all_points = pd.concat(frames, ignore_index=True)
    return _aggregate_with_ci_local(all_points, x_col='iteration', y_col='value')


def _plot_pref_eval_by_iterations(per_dir_dfs, pref_tag, eval_tag, total_tag, pref_count_tag, eval_count_tag, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    pref_agg = _aggregate_bald_by_iterations(per_dir_dfs, pref_tag, total_tag=total_tag, pref_count_tag=pref_count_tag, eval_count_tag=eval_count_tag)
    eval_agg = _aggregate_bald_by_iterations(per_dir_dfs, eval_tag, total_tag=total_tag, pref_count_tag=pref_count_tag, eval_count_tag=eval_count_tag)

    pref_color = '#E69F00'
    eval_color = '#4477AA'
    if pref_agg is not None and not pref_agg.empty:
        ax.plot(pref_agg['iteration'], pref_agg['mean_value'], color=pref_color, linewidth=2)
        ax.fill_between(pref_agg['iteration'], pref_agg['ci_low'], pref_agg['ci_high'], color=pref_color, alpha=0.20)
    if eval_agg is not None and not eval_agg.empty:
        ax.plot(eval_agg['iteration'], eval_agg['mean_value'], color=eval_color, linewidth=2)
        ax.fill_between(eval_agg['iteration'], eval_agg['ci_low'], eval_agg['ci_high'], color=eval_color, alpha=0.20)

    ax.set_xlabel('Active-Learning Iteration (derived from query updates)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    modality_handles = [
        Line2D([0], [0], color=pref_color, linestyle='-', linewidth=2, label='Preference'),
        Line2D([0], [0], color=eval_color, linestyle='-', linewidth=2, label='Evaluative')
    ]
    ax.legend(handles=modality_handles, title='Modality')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

# ---------------------------------------------
# Main CLI
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Unified Active Learning Visualization')
    parser.add_argument('log_dirs', nargs='+', help='One or more folders containing TensorBoard event subdirectories')
    # Diagram toggles
    parser.add_argument('--no-diagram3', action='store_true', help='Skip Diagram 3')
    parser.add_argument('--no-diagram4', action='store_true', help='Skip Diagram 4')
    parser.add_argument('--no-diagram5', action='store_true', help='Skip Diagram 5')
    parser.add_argument('--no-diagram6', action='store_true', help='Skip Diagram 6')
    parser.add_argument('--no-diagram7', action='store_true', help='Skip Diagram 7')
    parser.add_argument('--no-diagram8', action='store_true', help='Skip Diagram 8 (desired modality stacked bars)')
    parser.add_argument('--no-diagram10', action='store_true', help='Generate Diagram 10 (alternative desired modality plot)')

    # Older toggles retained for compatibility
    parser.add_argument('--no-modality', action='store_true', help='Skip modality stacked bar plot (Diagram 7)')
    parser.add_argument('--no-bald', action='store_true', help='Skip BALD distribution plot')
    parser.add_argument('--no-bald-timeseries', action='store_true', help='Skip OLD BALD time series plot')
    # Labels/colors
    parser.add_argument('--legend-names', nargs='*', default=None, help='Legend names per log dir (in order)')
    parser.add_argument('--colors', nargs='*', default=None, help='Colors per log dir (in order)')
    # Tags
    parser.add_argument('--pref-count-tag', default='params/num_train_pref_data', help='Preference count metric tag')
    parser.add_argument('--eval-count-tag', default='params/num_train_eval_data', help='Evaluative count metric tag')
    parser.add_argument('--total-count-tag', default='params/num_train_data', help='Total queries metric tag (if available)')
    # Normalized/Unnormalized BALD tags per modality
    parser.add_argument('--pref-bald-norm-tag', default='active_learning/avg_bald_score_bits', help='Normalized (bits) BALD for preferences')
    parser.add_argument('--eval-bald-norm-tag', default='active_learning/avg_evaluative_bald_score', help='Normalized BALD for evaluations')
    parser.add_argument('--pref-bald-raw-tag', default='active_learning/avg_bald_score_nats', help='Unnormalized (nats) BALD for preferences')
    parser.add_argument('--eval-bald-raw-tag', default='active_learning/avg_evaluative_bald_score_raw', help='Unnormalized BALD for evaluations')
    # Desired modality tags (adaptive quota)
    parser.add_argument('--desired-pref-tag', default='active_learning/adaptive_desired_pref', help='Desired preference count per iteration')
    parser.add_argument('--desired-eval-tag', default='active_learning/adaptive_desired_eval', help='Desired evaluative count per iteration')
    # Other
    parser.add_argument('--num-stages', type=int, default=5, help='Number of training stages for BALD distribution')
    parser.add_argument('--max-bars', type=int, default=None, help='Max number of bars (downsample iterations)')
    parser.add_argument('--output-dir', default='.', help='Directory to save plots')
    parser.add_argument('--x-iterations', dest='x_iterations', action='store_true', help='For Diagrams 3/4, plot vs AL iterations derived from query updates instead of raw timesteps')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve legend and colors
    legend_names = args.legend_names if args.legend_names and len(args.legend_names) == len(args.log_dirs) else None
    colors = args.colors if args.colors and len(args.colors) == len(args.log_dirs) else None

    # Build per-directory dfs maps with needed tags
    needed_tags = set()
    # Counts and totals for Diagram 7 and queries alignment
    needed_tags.update([args.pref_count_tag, args.eval_count_tag, args.total_count_tag])
    # BALD tags
    needed_tags.update([args.pref_bald_norm_tag, args.eval_bald_norm_tag, args.pref_bald_raw_tag, args.eval_bald_raw_tag])
    # Desired tags
    needed_tags.update([args.desired_pref_tag, args.desired_eval_tag])
    needed_tags = list(needed_tags)

    per_dir_dfs = []
    for log_dir in args.log_dirs:
        dfs = load_tensorboard_scalars(log_dir, tags=needed_tags)
        per_dir_dfs.append(dfs)

    # Diagram 3: Normalized BALD vs Timesteps (stage-aggregated) OR vs Iterations (derived)
    if not args.no_diagram3:
        if args.x_iterations:
            _plot_pref_eval_by_iterations(
                per_dir_dfs,
                pref_tag=args.pref_bald_norm_tag,
                eval_tag=args.eval_bald_norm_tag,
                total_tag=args.total_count_tag,
                pref_count_tag=args.pref_count_tag,
                eval_count_tag=args.eval_count_tag,
                ylabel='Normalized BALD Score',
                title='Normalized BALD Score vs Iterations (derived from query updates)',
                output_path=os.path.join(args.output_dir, 'diagram3_bald_norm_vs_iterations.png')
            )
        else:
            _plot_pref_eval_by_stages(
                per_dir_dfs,
                pref_tag=args.pref_bald_norm_tag,
                eval_tag=args.eval_bald_norm_tag,
                num_stages=args.num_stages,
                ylabel='Normalized BALD Score',
                title='Normalized BALD Score vs Timesteps (Stage-aggregated)',
                output_path=os.path.join(args.output_dir, 'diagram3_bald_norm_vs_steps.png')
            )

    # Diagram 4: Unnormalized BALD vs Timesteps (stage-aggregated) OR vs Iterations (derived)
    if not args.no_diagram4:
        if args.x_iterations:
            _plot_pref_eval_by_iterations(
                per_dir_dfs,
                pref_tag=args.pref_bald_raw_tag,
                eval_tag=args.eval_bald_raw_tag,
                total_tag=args.total_count_tag,
                pref_count_tag=args.pref_count_tag,
                eval_count_tag=args.eval_count_tag,
                ylabel='Unnormalized BALD Score',
                title='Unnormalized BALD Score vs Iterations (derived from query updates)',
                output_path=os.path.join(args.output_dir, 'diagram4_bald_raw_vs_iterations.png')
            )
        else:
            _plot_pref_eval_by_stages(
                per_dir_dfs,
                pref_tag=args.pref_bald_raw_tag,
                eval_tag=args.eval_bald_raw_tag,
                num_stages=args.num_stages,
                ylabel='Unnormalized BALD Score',
                title='Unnormalized BALD Score vs Timesteps (Stage-aggregated)',
                output_path=os.path.join(args.output_dir, 'diagram4_bald_raw_vs_steps.png')
            )

    # Diagram 5: Normalized BALD vs number of queries
    if not args.no_diagram5:
        series_by_dir = {}
        for i, dfs in enumerate(per_dir_dfs):
            label = legend_names[i] if legend_names else os.path.basename(os.path.normpath(args.log_dirs[i]))
            agg = _build_combined_bald_by_queries_per_dir(
                dfs,
                pref_tag=args.pref_bald_norm_tag,
                eval_tag=args.eval_bald_norm_tag,
                total_tag=args.total_count_tag,
                pref_count_tag=args.pref_count_tag,
                eval_count_tag=args.eval_count_tag,
            )
            series_by_dir[label] = agg
        _plot_multi_series(series_by_dir, x_col='queries', ylabel='Normalized BALD Score', title='Normalized BALD Score vs Number of Queries', output_path=os.path.join(args.output_dir, 'diagram5_bald_norm_vs_queries.png'), colors=colors)

    # Diagram 6: Unnormalized BALD vs number of queries
    if not args.no_diagram6:
        series_by_dir = {}
        for i, dfs in enumerate(per_dir_dfs):
            label = legend_names[i] if legend_names else os.path.basename(os.path.normpath(args.log_dirs[i]))
            agg = _build_combined_bald_by_queries_per_dir(
                dfs,
                pref_tag=args.pref_bald_raw_tag,
                eval_tag=args.eval_bald_raw_tag,
                total_tag=args.total_count_tag,
                pref_count_tag=args.pref_count_tag,
                eval_count_tag=args.eval_count_tag,
            )
            series_by_dir[label] = agg
        _plot_multi_series(series_by_dir, x_col='queries', ylabel='Unnormalized BALD Score', title='Unnormalized BALD Score vs Number of Queries', output_path=os.path.join(args.output_dir, 'diagram6_bald_raw_vs_queries.png'), colors=colors)

    # Diagram 7: Stacked bar for number of queries of each feedback type
    if not args.no_diagram7 and not args.no_modality:
        # Build stacked bars per directory and save separate figures to keep legend/colors mapping simple per dir.
        for i, dfs in enumerate(per_dir_dfs):
            label = legend_names[i] if legend_names else os.path.basename(os.path.normpath(args.log_dirs[i]))
            counts_df = build_modality_counts_dataframe(dfs, args.pref_count_tag, args.eval_count_tag)
            out_path = os.path.join(args.output_dir, f'diagram7_modality_counts_stacked_{label.replace(" ", "_")}.png')
            plot_modality_counts(counts_df, out_path, max_bars=args.max_bars)

    # Diagram 8: Stacked bar for desired queries of each feedback type per iteration
    if not args.no_diagram8:
        for i, dfs in enumerate(per_dir_dfs):
            label = legend_names[i] if legend_names else os.path.basename(os.path.normpath(args.log_dirs[i]))
            desired_df = build_desired_modality_dataframe(dfs, args.desired_pref_tag, args.desired_eval_tag)
            if desired_df is None or desired_df.empty:
                print(f"No desired modality data for {label}; skipping Diagram 8 for this dir.")
                continue
            out_path = os.path.join(args.output_dir, f'diagram8_desired_modality_stacked_{label.replace(" ", "_")}.png')
            plot_desired_modality_counts(desired_df, out_path, max_bars=args.max_bars)


    # Diagram 10: Alternative desired modality plot
    if not args.no_diagram10:
        # Assuming we use the first directory for this plot as well, like diagram 8
        all_desired = build_desired_modality_dataframe(per_dir_dfs[0], args.desired_pref_tag, args.desired_eval_tag)
        plot_desired_modality_counts_v2(all_desired,
                                        output_path=os.path.join(args.output_dir, 'diagram10_desired_modality_counts_v2.png'),
                                        max_bars=args.max_bars)

if __name__ == '__main__':
    main()
