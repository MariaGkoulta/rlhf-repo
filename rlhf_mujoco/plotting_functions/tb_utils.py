import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorBoard imports (shared across plotting scripts)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator  # type: ignore
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False


def load_tensorboard_scalars(log_dir, tags=None):
    """Recursively load scalar metrics from all TensorBoard event files under log_dir.

    Returns dict[tag] -> DataFrame(step, value, wall_time, file_path, run_id)
    """
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Please install tensorboard.")
        return {}

    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, f))
    if not event_files:
        print(f"No event files found under {log_dir}")
        return {}

    all_data = {}
    for ef in event_files:
        run_id = os.path.relpath(os.path.dirname(ef), log_dir)
        try:
            if 'EventAccumulator' in globals():
                ea = EventAccumulator(ef)
                ea.Reload()
                scalar_tags = ea.Tags().get('scalars', [])
                for tag in scalar_tags:
                    if tags is not None and tag not in tags:
                        continue
                    if tag not in all_data:
                        all_data[tag] = []
                    for ev in ea.Scalars(tag):
                        all_data[tag].append({
                            'step': ev.step,
                            'value': ev.value,
                            'wall_time': ev.wall_time,
                            'file_path': ef,
                            'run_id': run_id
                        })
            else:
                from tensorflow.python.summary.summary_iterator import summary_iterator  # type: ignore
                for event in summary_iterator(ef):
                    if not getattr(event, 'summary', None) or not event.summary.value:
                        continue
                    for val in event.summary.value:
                        tag = getattr(val, 'tag', None)
                        if tag is None:
                            continue
                        if tags is not None and tag not in tags:
                            continue
                        # Support both TF proto styles
                        has_simple = False
                        simple_value = None
                        try:
                            has_simple = val.HasField('simple_value')  # type: ignore[attr-defined]
                            simple_value = val.simple_value  # type: ignore[attr-defined]
                        except Exception:
                            simple_value = getattr(val, 'simple_value', None)
                            has_simple = simple_value is not None
                        if not has_simple:
                            continue
                        if tag not in all_data:
                            all_data[tag] = []
                        all_data[tag].append({
                            'step': getattr(event, 'step', np.nan),
                            'value': simple_value,
                            'wall_time': getattr(event, 'wall_time', np.nan),
                            'file_path': ef,
                            'run_id': run_id
                        })
        except Exception as e:
            print(f"Error reading {ef}: {e}")
            continue

    dfs = {}
    for tag, rows in all_data.items():
        if rows:
            df = pd.DataFrame(rows).sort_values('step').reset_index(drop=True)
            dfs[tag] = df
    return dfs


def _aggregate_with_ci(df, value_col='value', group_col='step'):
    """Helper: aggregate mean/std/count by group_col and compute 95% CI.

    Returns DataFrame with columns: group_col, mean_value, ci_low, ci_high, n.
    """
    grouped = df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
    grouped.rename(columns={'mean': 'mean_value', 'count': 'n'}, inplace=True)
    grouped['std'] = grouped['std'].fillna(0.0)
    grouped['stderr'] = grouped.apply(lambda r: (r['std'] / np.sqrt(r['n'])) if r['n'] > 0 else 0.0, axis=1)
    grouped['ci'] = 1.96 * grouped['stderr']
    grouped['ci_low'] = grouped['mean_value'] - grouped['ci']
    grouped['ci_high'] = grouped['mean_value'] + grouped['ci']
    return grouped[[group_col, 'mean_value', 'ci_low', 'ci_high', 'n']]


def build_bald_time_series(dfs, pref_bald_tag, eval_bald_tag):
    """Aggregate BALD scores across runs per training step for each modality.

    Returns DataFrame with columns: modality, step, mean_value, ci_low, ci_high, n.
    """
    frames = []
    for tag, modality in [(pref_bald_tag, 'preference'), (eval_bald_tag, 'evaluative')]:
        if not tag or tag not in dfs:
            continue
        d = dfs[tag].copy()
        agg = _aggregate_with_ci(d, value_col='value', group_col='step')
        agg['modality'] = modality
        frames.append(agg[['modality', 'step', 'mean_value', 'ci_low', 'ci_high', 'n']])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(['modality', 'step'])


def build_bald_time_series_multi(dfs, tag_specs):
    """Aggregate multiple series (e.g., different BALD variants) across runs per step.

    tag_specs: list of tuples (tag, label). Missing tags are skipped.
    Returns DataFrame columns: series, step, mean_value, ci_low, ci_high, n
    """
    frames = []
    for tag, label in tag_specs:
        if not tag or tag not in dfs:
            print(f"Tag not found, skipping: {tag}")
            continue
        d = dfs[tag].copy()
        agg = _aggregate_with_ci(d, value_col='value', group_col='step')
        agg['series'] = label
        frames.append(agg[['series', 'step', 'mean_value', 'ci_low', 'ci_high', 'n']])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(['series', 'step'])


# -------------------------------------------------------------
# Queries-based aggregation and plotting
# -------------------------------------------------------------

def build_time_series_by_queries(dfs, tag_specs, query_tag='params/num_train_data'):
    """Aggregate one or more scalar series across runs using number of queries on the x-axis.

    Inputs
    - dfs: dict[tag] -> DataFrame(step, value, wall_time, file_path, run_id)
    - tag_specs: list of (tag, label) pairs to include as separate series
    - query_tag: tag name that provides the number of queries over steps (default: 'params/num_train_data')

    Output
    - DataFrame with columns: series, queries, mean_value, ci_low, ci_high, n

    Notes
    - For each run_id, metric values are aligned to the most recent queries count at or before each step
      (merge_asof with direction='backward'). Multiple metric points mapping to the same queries value per run
      are averaged before cross-run aggregation.
    """
    if query_tag not in dfs:
        print(f"Query tag not found: {query_tag}")
        return pd.DataFrame()

    # Prepare per-run queries timelines
    qdf = dfs[query_tag].copy()
    qdf = qdf[['run_id', 'step', 'value']].rename(columns={'value': 'queries'})
    qdf = qdf.sort_values(['run_id', 'step'])
    run_ids_with_queries = set(qdf['run_id'].unique())

    frames = []
    for tag, label in tag_specs:
        if not tag or tag not in dfs:
            print(f"Skipping missing tag: {tag}")
            continue
        mdf = dfs[tag][['run_id', 'step', 'value']].copy().sort_values(['run_id', 'step'])
        run_ids_metrics = set(mdf['run_id'].unique())
        run_ids = sorted(run_ids_with_queries.intersection(run_ids_metrics))
        if not run_ids:
            continue

        for rid in run_ids:
            m_run = mdf[mdf['run_id'] == rid][['step', 'value']].reset_index(drop=True)
            q_run = qdf[qdf['run_id'] == rid][['step', 'queries']].reset_index(drop=True)
            if m_run.empty or q_run.empty:
                continue
            # Align each metric point to latest known queries count at or before that step
            aligned = pd.merge_asof(m_run.sort_values('step'), q_run.sort_values('step'), on='step', direction='backward')
            aligned = aligned.dropna(subset=['queries'])
            if aligned.empty:
                continue
            # Average duplicates per queries to get single point per queries per run
            per_q = aligned.groupby('queries', as_index=False)['value'].mean()
            per_q['series'] = label
            per_q['run_id'] = rid
            frames.append(per_q[['series', 'run_id', 'queries', 'value']])

    if not frames:
        return pd.DataFrame()

    all_points = pd.concat(frames, ignore_index=True)
    # Aggregate across runs for each (series, queries)
    agg = all_points.groupby(['series', 'queries']).agg(
        mean_value=('value', 'mean'),
        std=('value', 'std'),
        n=('value', 'count')
    ).reset_index()
    agg['std'] = agg['std'].fillna(0.0)
    agg['stderr'] = agg.apply(lambda r: (r['std'] / np.sqrt(r['n'])) if r['n'] > 0 else 0.0, axis=1)
    agg['ci'] = 1.96 * agg['stderr']
    agg['ci_low'] = agg['mean_value'] - agg['ci']
    agg['ci_high'] = agg['mean_value'] + agg['ci']
    return agg[['series', 'queries', 'mean_value', 'ci_low', 'ci_high', 'n']].sort_values(['series', 'queries'])


def plot_time_series_by_queries(ts_df, output_path, ylabel='Metric Value', title='Metric vs Number of Queries', series_order=None, colors=None):
    """Plot mean ± 95% CI for one or more series against number of queries.

    Inputs
    - ts_df: DataFrame from build_time_series_by_queries with columns
             [series, queries, mean_value, ci_low, ci_high, n]
    - output_path: file path to save the plot (png/pdf, etc.)
    - ylabel: y-axis label
    - title: plot title
    - series_order: optional list specifying the order of series in the legend
    - colors: optional dict {series_label: color}
    """
    if ts_df is None or ts_df.empty:
        print("No data to plot (queries-based time series is empty).")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    default_palette = [
        '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#1f77b4', '#2ca02c'
    ]
    colors = colors or {}

    # Determine plotting order
    series_list = list(ts_df['series'].unique())
    if series_order:
        # Preserve only those present and in given order, then append the rest
        ordered = [s for s in series_order if s in series_list]
        ordered += [s for s in series_list if s not in ordered]
        series_list = ordered

    for i, series in enumerate(series_list):
        sub = ts_df[ts_df['series'] == series].sort_values('queries')
        color = colors.get(series, default_palette[i % len(default_palette)])
        ax.plot(sub['queries'], sub['mean_value'], label=series, color=color, linewidth=2)
        ax.fill_between(sub['queries'], sub['ci_low'], sub['ci_high'], color=color, alpha=0.2)

    ax.set_xlabel('Number of Queries')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved queries-based time series plot to {output_path}")
    plt.close(fig)
