"""Comparing a nyx hypnogram against a reference (manually scored) hypnogram."""


import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from nyx.scoring import prepare_epoch_data


def expand_manual_labels(manual_labels, epoch_length, fs, label_map = None):
    """
    Expand manual epoch labels to sample-level labels.
    
    Parameters
    ----------
    manual_labels : array-like
        1D array of labels per epoch (e.g., [0,1,2,...]).
    epoch_length : float
        Duration of each epoch in seconds.
    fs : float
        Sampling frequency (samples per second).
    label_map : dict, optional
        
    Returns
    -------
    expanded_labels : : dict
        {
            'time': array of segment start times (s),
            'duration': array of segment durations (s),
            'label': array of segment labels
        }
    """
    manual_labels = np.asarray(manual_labels)
    samples_per_epoch = int(round(epoch_length * fs))
    expanded_labels = np.repeat(manual_labels, samples_per_epoch)
    return prepare_epoch_data(expanded_labels, fs, state_names = label_map if label_map else {})


def merge_states(data: dict, merge_map: dict) -> dict:
    """
    Merge specified states into a target state and combine consecutive same states.

    Parameters:
        data (dict): Input data with 'time', 'duration', and 'label' keys.
        merge_map (dict): Mapping of states to merge {source_state: target_state}.

    Returns:
        dict: Updated data with merged states and combined consecutive same states.
    """
    if not all(key in data for key in ['time', 'duration', 'label']):
        raise KeyError("Input data must contain 'time', 'duration', and 'label' keys.")

    merged_time, merged_duration, merged_label = [], [], []
    current_label = None
    current_start = None
    current_duration = 0

    for start, duration, label in zip(data['time'], data['duration'], data['label']):
        # Map the label to the target state if in merge_map
        label = merge_map.get(label, label)

        if label == current_label:
            # Extend the current segment
            current_duration += duration
        else:
            # Save the previous segment
            if current_label is not None:
                merged_time.append(current_start)
                merged_duration.append(current_duration)
                merged_label.append(current_label)
            # Start a new segment
            current_label = label
            current_start = start
            current_duration = duration

    # Save the last segment
    if current_label is not None:
        merged_time.append(current_start)
        merged_duration.append(current_duration)
        merged_label.append(current_label)

    return {
        'time': np.array(merged_time, dtype='float64'),
        'duration': np.array(merged_duration, dtype='float64'),
        'label': np.array(merged_label, dtype='U'),
    }
    
def trim_manual_scores(manual_scores_dict, time_start, time_end):
    """
    Trim manual scores by slicing to time range.
    """
    df = pd.DataFrame(manual_scores_dict)
    
    # Filter by time range
    df = df[(df['time'] < time_end) & (df['time'] + df['duration'] > time_start)].reset_index(drop=True)
    
    # Adjust start time/duration of the first epoch if it starts before time_start
    if not df.empty:
        start_diff = time_start - df.loc[0, 'time']
        if start_diff > 0:
            df.loc[0, 'time'] = time_start
            df.loc[0, 'duration'] -= start_diff

        # Adjust duration of the last epoch if it ends after time_end
        last_idx = df.index[-1]
        end_time = df.loc[last_idx, 'time'] + df.loc[last_idx, 'duration']
        end_diff = end_time - time_end
        if end_diff > 0:
            df.loc[last_idx, 'duration'] -= end_diff
            
        # Shift time to start from 0 relative to time_start
        df['time'] = df['time'] - time_start

    # Convert back to dict
    refined_dict = df.to_dict(orient='list')
    
    return refined_dict

def _to_df(data):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        raise ValueError("data must be a dict or pandas.DataFrame")
    if not set(['time','duration','label']).issubset(df.columns):
        raise ValueError("data must have 'time','duration','label' columns")
    if np.issubdtype(df['time'].dtype, np.number):
        df['time'] = pd.to_datetime(df['time'], unit='s')
    else:
        df['time'] = pd.to_datetime(df['time'])
    if np.issubdtype(df['duration'].dtype, np.timedelta64):
        df['duration'] = df['duration'].dt.total_seconds()
    else:
        df['duration'] = pd.to_numeric(df['duration']).astype(float)
    df = df.sort_values('time').reset_index(drop=True)
    return df

def _map_annotations_to_epochs(df, epochs):
    """Maps annotations to a fixed epoch grid using a majority vote based on duration."""
    if len(df) == 0:
        return np.array([np.nan] * len(epochs), dtype=object)

    epoch_len = (epochs[1] - epochs[0]) if len(epochs) > 1 else (epochs[0] - epochs[0])
    if len(epochs) == 1:
        # Estimate epoch length from data if only one epoch is given
        epoch_len = pd.to_timedelta(df['duration'].median(), unit='s')

    out_labels = []
    df_starts = df['time']
    df_ends = df['time'] + pd.to_timedelta(df['duration'], unit='s')
    df_labels = df['label']

    for epoch_start in epochs:
        epoch_end = epoch_start + epoch_len
        
        # Find segments that overlap with the current epoch
        overlapping_indices = (df_starts < epoch_end) & (df_ends > epoch_start)
        
        if not overlapping_indices.any():
            out_labels.append(np.nan)
            continue

        overlapping_segments = df[overlapping_indices]
        
        # Calculate duration of each label within the epoch
        label_durations = {}
        for _, seg in overlapping_segments.iterrows():
            overlap_start = max(epoch_start, seg['time'])
            overlap_end = min(epoch_end, seg['time'] + pd.to_timedelta(seg['duration'], unit='s'))
            duration = (overlap_end - overlap_start).total_seconds()
            
            label = seg['label']
            label_durations[label] = label_durations.get(label, 0) + duration
            
        # Find the label with the maximum duration (majority vote)
        if not label_durations:
            out_labels.append(np.nan)
        else:
            majority_label = max(label_durations, key=label_durations.get)
            out_labels.append(majority_label)
            
    return np.array(out_labels, dtype=object)

# Aggregator mapping to WAKE/REM/NREM
def _aggregate_label(lbl):
    if pd.isna(lbl):
        return np.nan
    s = str(lbl).strip().upper()
    # Wake variants
    if s in ('WAKE','W','AWAKE', 'AWAKE (ARTEFACT)', 'SLEEP MOVEMENT', 'WAKE ', ' WAKE X'):
        return 'WAKE'
    if s in ('QW','QUIETWAKE'):
        return 'QW'
    # REM variants
    if s in ('REM','R', 'REM (ARTEFACT)', 'REM X', 'REM '):
        return 'REM'
    # NREM variants
    if s in ('NREM1','N1', 'IS N-R', 'IS R-N'):
        return 'NREM1'
    if s in ('NREM2','N2'):
        return 'NREM2'
    if s in ('NREM3','N3'):
        return 'NREM3'
    
    # NREM: N1,N2,N3,NREM, NREM variants, 'N' etc.
    if s in ('NREM','NREM_SLEEP','NON-REM','NONREM', 'NON-REM (ARTEFACT)', 'N', 'NREM X', 'NREM', 'NON REM'):
        return 'NREM'
    # Some label sets use 'S' for sleep or 'ASLEEP' - map to NREM by default
    if s in ('S','ASLEEP','SLEEP'):
        return 'SLEEP'
    # fallback: if contains 'REM' anywhere -> REM, if contains 'W' or 'WA' -> WAKE, else NREM
    if 'REM' in s:
        return 'REM'
    if 'W' == s or 'WAKE' in s:
        return 'WAKE'
    # default to NREM
    return 'NREM'

def compare_sleep(auto_data, manual_data, label_order = ['WAKE','QW','REM','NREM'], start=None, end=None, normalize_cm=False, plot=True, debug=False, verbose=True):
    """
    Compare automatic and manual scoring using a fine-grained, event-based approach.
    Labels are first aggregated to WAKE, REM, NREM, etc.
    The confusion matrix and classification report are printed in the specified label_order.
    """
    auto_df = _to_df(auto_data)
    manual_df = _to_df(manual_data)

    # --- Create the fine-grained comparison grid ---
    all_times = pd.concat([
        auto_df['time'], auto_df['time'] + pd.to_timedelta(auto_df['duration'], unit='s'),
        manual_df['time'], manual_df['time'] + pd.to_timedelta(manual_df['duration'], unit='s')
    ]).unique()
    
    # Filter by start/end if provided
    data_start = min(auto_df['time'].min(), manual_df['time'].min())
    data_end = max((auto_df['time'] + pd.to_timedelta(auto_df['duration'], unit='s')).max(),
                   (manual_df['time'] + pd.to_timedelta(manual_df['duration'], unit='s')).max())
    start_ts = pd.to_datetime(start, unit='s') if start is not None else data_start
    end_ts = pd.to_datetime(end, unit='s') if end is not None else data_end
    
    all_times = sorted([t for t in all_times if start_ts <= t <= end_ts])
    
    if len(all_times) < 2:
        print("Warning: Not enough overlapping data to create comparison grid.")
        return None, None, None

    # --- Map labels to the fine-grained grid ---
    y_auto, y_manual, durations = [], [], []
    # Debug instrumentation containers
    manual_grid_durations = {}
    manual_overlap_durations = {}
    skipped_manual_due_to_no_auto = 0.0
    skipped_due_to_no_manual = 0.0
    for i in range(len(all_times) - 1):
        t_start, t_end = all_times[i], all_times[i+1]
        t_mid = t_start + (t_end - t_start) / 2
        duration = (t_end - t_start).total_seconds()

        if duration < 1e-6: continue

        # Find labels at the midpoint of the micro-epoch
        auto_label = auto_df[(auto_df['time'] <= t_mid) & ((auto_df['time'] + pd.to_timedelta(auto_df['duration'], unit='s')) > t_mid)]['label'].values
        manual_label = manual_df[(manual_df['time'] <= t_mid) & ((manual_df['time'] + pd.to_timedelta(manual_df['duration'], unit='s')) > t_mid)]['label'].values

        # Track manual label coverage across the grid (independent of auto)
        if len(manual_label) > 0:
            lbl = manual_label[0]
            manual_grid_durations[lbl] = manual_grid_durations.get(lbl, 0.0) + duration
        else:
            skipped_due_to_no_manual += duration

        if len(manual_label) > 0 and len(auto_label) == 0:
            skipped_manual_due_to_no_auto += duration

        if len(auto_label) > 0 and len(manual_label) > 0:
            y_auto.append(auto_label[0])
            y_manual.append(manual_label[0])
            durations.append(duration)
            # Track overlapped manual durations
            lbl = manual_label[0]
            manual_overlap_durations[lbl] = manual_overlap_durations.get(lbl, 0.0) + duration

    if not y_auto:
        print("Warning: No overlapping labeled epochs between automatic and manual data.")
        return None, None, None

    # --- Aggregate labels and calculate metrics ---
    y_auto_ag = np.array([_aggregate_label(x) for x in y_auto], dtype=object)
    y_manual_ag = np.array([_aggregate_label(x) for x in y_manual], dtype=object)
    
    # Use durations as sample weights for metrics
    sample_weights = np.array(durations)

    # fixed order for metrics
    cm = confusion_matrix(y_manual_ag, y_auto_ag, labels=label_order, sample_weight=sample_weights)
    acc = accuracy_score(y_manual_ag, y_auto_ag, sample_weight=sample_weights)
    kappa = cohen_kappa_score(y_manual_ag, y_auto_ag, sample_weight=sample_weights)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_manual_ag, y_auto_ag, labels=label_order, zero_division=0, sample_weight=sample_weights
    )
    
    # Re-calculate support as sum of durations for each true label
    support_by_duration = {label: 0 for label in label_order}
    for label, weight in zip(y_manual_ag, sample_weights):
        if label in support_by_duration:
            support_by_duration[label] += weight
    supports_dur = np.array([support_by_duration[label] for label in label_order])

    metrics_df = pd.DataFrame({
        'precision': precisions,
        'recall': recalls,
        'f1-score': f1s,
        'support': supports_dur
    }, index=label_order)

    total_support = metrics_df['support'].sum()

    # Macro (unweighted) averages. A weighted average lets the dominant stage
    # hide poor performance on a rare one -- REM is a small fraction of a
    # recording, so weighting by support makes missing it look cheap. MF1, the
    # macro F1, is the headline number.
    #
    # Averaged over the stages actually present in the reference: scoring a
    # recording that contains no REM should not be penalised for a stage that
    # was never there to find.
    present = supports_dur > 0
    if present.any():
        metrics_overall = pd.Series({
            'precision': float(np.mean(precisions[present])),
            'recall': float(np.mean(recalls[present])),
            'f1-score': float(np.mean(f1s[present])),
            'support': total_support,
        }, name='macro avg')
    else:
        metrics_overall = pd.Series(
            {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0.0},
            name='macro avg',
        )

    results = {
        'y_manual': y_manual_ag,
        'y_auto': y_auto_ag,
        'epochs': all_times[:-1],
        'durations': durations,
        'label_order': label_order,
        'confusion_matrix': cm,
        'metrics_df': metrics_df,
        'accuracy': float(acc),
        'kappa': float(kappa),
        'mf1': float(metrics_overall['f1-score']),
    }

    # Plotting is complex with non-uniform epochs, so it's disabled for now.
    if plot:
        print("Plotting is not supported for fine-grained comparison yet.")

    # print nicely
    if verbose:
        print("=== Summary metrics (aggregated, fine-grained comparison) ===")
        print(f"MF1: {results['mf1']:.3f}    Accuracy: {results['accuracy']:.3f}    "
              f"Cohen's kappa: {results['kappa']:.3f}\n")
        print(metrics_df.to_string(formatters={"precision": "{:.3f}".format,
                                               "recall": "{:.3f}".format,
                                               "f1-score": "{:.3f}".format,
                                               "support": "{:.1f}s".format}))
        print("\nMacro average (unweighted, over stages present in the reference):")
        print(metrics_overall.to_frame().T.map(lambda x: f"{x:.3f}" if isinstance(x, (float,np.floating)) else x))
        print("\nConfusion matrix (rows=true, cols=predicted, values are in seconds):")
        print(pd.DataFrame(cm, index=label_order, columns=label_order).round(1).to_string())

    if debug:
        # Provide visibility into manual labels present and how much got included/excluded
        print("\n=== Debug: Manual label coverage ===")
        # Total manual durations by label from original segments
        total_manual_by_label = manual_df.groupby('label')['duration'].sum()
        print("Total manual durations by label (segment sums, seconds):")
        print(total_manual_by_label.to_string())

        print("\nManual durations by label on fine grid (seconds):")
        print(pd.Series(manual_grid_durations).sort_index().to_string())

        print("\nManual durations included in overlap (seconds):")
        print(pd.Series(manual_overlap_durations).sort_index().to_string())

        print(f"\nManual duration skipped due to no auto overlap: {skipped_manual_due_to_no_auto:.2f}s")
        print(f"Manual duration with no manual label at midpoints (gaps): {skipped_due_to_no_manual:.2f}s")

    return results, metrics_df, metrics_overall

