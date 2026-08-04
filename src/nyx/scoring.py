"""Turning feature traces into hypnograms, and reading/writing hypnograms."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from nyx.stages import NOSIGNAL, NREM, REM, SLEEP, STATE_NAMES, UNCLASSIFIED, WAKE

# Kept as a module-level name because several functions below take it as a
# default argument.
state_names: dict[int, str] = STATE_NAMES


# Utility functions
def _segments(starts_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return start and end indices (vectorized) for a boolean mask of a state."""
    changes = np.diff(np.concatenate(([0], starts_mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return starts, ends


def _validate_thresholds(thresholds: Iterable[float], expected_length: int = 2) -> list[float]:
    """Validate and ensure thresholds are of the correct length."""
    thr = list(thresholds)
    if len(thr) != expected_length:
        raise ValueError(f"thresholds must be a {expected_length}-tuple")
    return thr


def _reclassify_short_segments(classifications: np.ndarray, mask: np.ndarray, target_label: int, min_duration_samples: int):
    """Reclassify short segments of a given mask to a target label."""
    starts, ends = _segments(mask)
    for s, e in zip(starts, ends):
        if (e - s) < min_duration_samples:
            classifications[s:e] = target_label


# Main classification functions
def classify_wakesleep(power_emg_h: np.ndarray, thresholds: Iterable[float], fs: float, min_duration: float = 5) -> dict[str, np.ndarray]:
    """Classify WAKE/SLEEP (+ NOSIGNAL) from EMG power."""
    threshold_nosignal, threshold_emg = _validate_thresholds(thresholds)
    min_duration_samples = int(min_duration * fs)

    classifications = np.full(len(power_emg_h), UNCLASSIFIED, dtype=int)

    # Classify NOSIGNAL
    nosignal_indices = power_emg_h < threshold_nosignal
    classifications[nosignal_indices] = NOSIGNAL

    # Reclassify short UNCLASSIFIED as NOSIGNAL
    _reclassify_short_segments(classifications, ~nosignal_indices, NOSIGNAL, min_duration_samples)

    # Classify WAKE and SLEEP
    wake_indices = (classifications == UNCLASSIFIED) & (power_emg_h > threshold_emg)
    sleep_indices = (classifications == UNCLASSIFIED) & ~wake_indices
    classifications[wake_indices] = WAKE
    classifications[sleep_indices] = SLEEP

    # Reclassify short SLEEP -> WAKE and WAKE -> SLEEP
    _reclassify_short_segments(classifications, sleep_indices, WAKE, min_duration_samples)
    _reclassify_short_segments(classifications, classifications == WAKE, SLEEP, min_duration_samples)

    return prepare_epoch_data(classifications, fs)


def classify_sleep(feature_trace: np.ndarray, thresholds: Iterable[float] | float, wakesleep_hypno: dict[str, np.ndarray], fs: float, min_duration: float = 5) -> dict[str, np.ndarray]:
    """Within SLEEP segments, split into REM/NREM by a threshold on a feature trace (e.g., PC1)."""
    thr = float(np.atleast_1d(thresholds)[0])
    wake_sleep_df = pd.DataFrame(wakesleep_hypno)
    min_duration_samples = int(min_duration * fs)

    epoch_times, epoch_durations, epoch_labels = [], [], []

    for _, row in wake_sleep_df.iterrows():
        start_time, duration, label = row["time"], row["duration"], row["label"]
        end_time = start_time + duration

        start_idx = int(np.floor(start_time * fs))
        end_idx = int(np.floor(end_time * fs))

        if label == "SLEEP":
            # Skip if the segment has zero length in samples
            if start_idx >= end_idx:
                continue

            segment_feature = feature_trace[start_idx:end_idx]
            segment_time = np.arange(start_idx, end_idx) / fs

            rem_indices = segment_feature < thr
            nrem_indices = ~rem_indices

            seg_cls = np.full_like(segment_feature, UNCLASSIFIED, dtype=int)
            seg_cls[rem_indices] = REM
            seg_cls[nrem_indices] = NREM

            # Reclassify short REM -> NREM and NREM -> REM
            _reclassify_short_segments(seg_cls, rem_indices, NREM, min_duration_samples)
            _reclassify_short_segments(seg_cls, seg_cls == NREM, REM, min_duration_samples)

            # Collapse back to epochs
            current_label = seg_cls[0]
            current_start_time = float(segment_time[0])
            for i in range(1, len(seg_cls)):
                if seg_cls[i] != current_label:
                    current_duration = float(segment_time[i] - current_start_time)
                    epoch_times.append(current_start_time)
                    epoch_durations.append(current_duration)
                    epoch_labels.append(state_names[current_label])
                    current_start_time = float(segment_time[i])
                    current_label = int(seg_cls[i])
            final_duration = float(segment_time[-1] - current_start_time + (1 / fs))
            epoch_times.append(current_start_time)
            epoch_durations.append(final_duration)
            epoch_labels.append(state_names[current_label])
        else:
            epoch_times.append(start_time)
            epoch_durations.append(duration)
            epoch_labels.append(str(label))

    return {
        "time": np.array(epoch_times, dtype="float64"),
        "duration": np.array(epoch_durations, dtype="float64"),
        "label": np.array(epoch_labels, dtype="U"),
    }


# Utility for preparing epoch data
def prepare_epoch_data(classifications: np.ndarray, fs: float, state_names: dict[int, str] = state_names) -> dict[str, np.ndarray]:
    """Convert classification data into an epoch format."""
    if classifications.size == 0:
        return {
            "time": np.array([], dtype="float64"),
            "duration": np.array([], dtype="float64"),
            "label": np.array([], dtype="U"),
        }
    epoch_times, epoch_durations, epoch_labels = [], [], []

    current_state = classifications[0]
    start_time = 0

    for i in range(1, len(classifications)):
        time = i / fs
        if classifications[i] != current_state:
            end_time = time
            duration = end_time - start_time
            epoch_times.append(start_time)
            epoch_durations.append(duration)
            epoch_labels.append(state_names[current_state])
            current_state = classifications[i]
            start_time = end_time

    # Add the last epoch
    end_time = len(classifications) / fs
    duration = end_time - start_time
    epoch_times.append(start_time)
    epoch_durations.append(duration)
    epoch_labels.append(state_names[current_state])

    return {
        "time": np.array(epoch_times, dtype="float64"),
        "duration": np.array(epoch_durations, dtype="float64"),
        "label": np.array(epoch_labels, dtype="U"),
    }
    
def _to_epoch_df(epoch_data: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "time": epoch_data["time"],
        "duration": epoch_data["duration"],
        "label": epoch_data["label"],
    })


def savehypno(epoch_data, savepath):
    """Save hypnogram data to a CSV file."""           
    df = pd.DataFrame()
    df['time'] = np.round(epoch_data["time"], 6)         # round to nearest microsecond
    df['duration'] = np.round(epoch_data["duration"], 6) # round to nearest microsecond
    df['label'] = epoch_data["label"]
    df.sort_values(['time', 'duration', 'label'], inplace=True)
    df.to_csv(savepath, index=False)


def apply_forbidden_transitions(hypnogram):
    """
    Corrects forbidden transitions in a hypnogram.
    For example, a REM epoch surrounded by WAKE is converted to WAKE.
    It also merges consecutive states with the same label.
    """
    hypno_df = pd.DataFrame(hypnogram)
    if len(hypno_df) < 3:
        return hypnogram

    # Apply rule: WAKE -> REM -> WAKE becomes WAKE -> WAKE -> WAKE
    for i in range(1, len(hypno_df) - 1):
        prev_state = hypno_df.loc[i - 1, 'label']
        current_state = hypno_df.loc[i, 'label']
        next_state = hypno_df.loc[i + 1, 'label']

        if prev_state == 'WAKE' and current_state == 'REM' and next_state == 'WAKE':
            hypno_df.loc[i, 'label'] = 'WAKE'

    # Merge consecutive states with the same label
    merged_hypno = []
    if len(hypno_df) > 0:
        current_state = hypno_df.iloc[0].to_dict()
        for i in range(1, len(hypno_df)):
            next_state = hypno_df.iloc[i]
            if next_state['label'] == current_state['label']:
                current_state['duration'] += next_state['duration']
            else:
                merged_hypno.append(current_state)
                current_state = next_state.to_dict()
        merged_hypno.append(current_state)

    if not merged_hypno:
        return {'time': [], 'duration': [], 'label': []}

    # Convert list of dicts back to dict of lists
    return {
        'time': [d['time'] for d in merged_hypno],
        'duration': [d['duration'] for d in merged_hypno],
        'label': [d['label'] for d in merged_hypno]
    }
    
def save_hypno_with_padding(hypnogram, savepath, time_start, time_end, total_duration):
    """
    Saves a hypnogram with NOSIGNAL padding for the excluded periods.
    """
    import copy
    
    # Create a deep copy to avoid modifying the original dictionary in memory
    padded_hypno = copy.deepcopy(hypnogram)
    
    # Convert to lists for easier manipulation if they are numpy arrays
    times = list(padded_hypno['time'])
    durations = list(padded_hypno['duration'])
    labels = list(padded_hypno['label'])
    
    # Shift existing times
    times = [t + time_start for t in times]
    
    # Prepend NOSIGNAL if needed
    if time_start > 0:
        times.insert(0, 0.0)
        durations.insert(0, float(time_start))
        labels.insert(0, 'NOSIGNAL')
        
    # Append NOSIGNAL if needed
    if total_duration > time_end:
        times.append(float(time_end))
        durations.append(float(total_duration - time_end))
        labels.append('NOSIGNAL')
        
    padded_hypno['time'] = np.array(times)
    padded_hypno['duration'] = np.array(durations)
    padded_hypno['label'] = np.array(labels)
    
    savehypno(padded_hypno, savepath)
    print(f"Hypnogram saved to {savepath} with NOSIGNAL padding.")