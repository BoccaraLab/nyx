"""PCA of the EEG spectrogram, fitted on selected stages only."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def _preprocess_pca_input(powers_eeg_temp: np.ndarray) -> np.ndarray:
    """Normalize and reshape the input for PCA."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(powers_eeg_temp.reshape(-1, 1)).flatten().reshape(powers_eeg_temp.shape)

def run_pca(powers, fs, wakesleep_hypno=None, label=None, n_components=4):
    # Normalize using the function from scalogram.py
    wake_sleep_df = pd.DataFrame(wakesleep_hypno)
    
    if label is not None and wake_sleep_df is not None:
        # Ensure label is a list for consistent processing
        if not isinstance(label, list):
            labels_to_use = [label]
        else:
            labels_to_use = label

        # Concatenate all segments of the specified stages (labels)
        all_powers = []
        for _, row in wake_sleep_df.iterrows():
            if row['label'] in labels_to_use:
                start_idx = int(row['time'] * fs)
                end_idx = int((row['time'] + row['duration']) * fs)
                EEG_segment = powers[:, start_idx:end_idx]
                EEG_segment = np.where(EEG_segment == np.inf, 0., EEG_segment)
                all_powers.append(EEG_segment)

        if not all_powers:
            # Handle case where no segments match the label(s)
            powers_combined = np.array([]).reshape(powers.shape[0], 0)
        else:
            powers_combined = np.concatenate(all_powers, axis=1)
    else:
        powers_combined = powers

    # If no data to process, return early
    if powers_combined.shape[1] == 0:
        pca = PCA(n_components=n_components)
        # Fit on a dummy array to have a valid pca object with components_
        pca.fit(np.zeros((n_components, powers.shape[0])))
        pca_signal = np.full((powers.shape[1], n_components), np.nan, dtype=np.float16)
        return pca, np.array([]).reshape(0, n_components), pca_signal

    powers_transposed = powers_combined.T
    powers_transposed = _preprocess_pca_input(powers_transposed)

    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(powers_transposed)

    # Map it back to the signal
    pca_signal = np.full((powers.shape[1], n_components), np.nan)

    if label is not None and wake_sleep_df is not None:
        if not isinstance(label, list):
            labels_to_use = [label]
        else:
            labels_to_use = label
            
        # Process each sleep segment and map PCA results
        pca_pointer = 0  # Pointer to track where we are in the PCA results
        for _, row in wake_sleep_df.iterrows():
            if row['label'] in labels_to_use:
                start_idx = int(row['time'] * fs)
                end_idx = int((row['time'] + row['duration']) * fs)

                # Get the length of the current sleep segment
                segment_length = end_idx - start_idx

                # Extract the portion of the PCA results corresponding to this segment
                pca_segment = pca_result[pca_pointer:pca_pointer + segment_length, :]

                # Map this segment back to its position in the full-length array
                pca_signal[start_idx:end_idx, :] = pca_segment

                # Advance the PCA result pointer
                pca_pointer += segment_length

        pca_signal = pca_signal.astype(np.float16)

    return pca, pca_result, pca_signal


