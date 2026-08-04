"""PCA of the EEG spectrogram, fitted on selected stages only."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, peak_prominences
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

def orient_pca_by_peak(freqs, components, band=[1, 20], components_to_use=[0,1], plot=False, save_path=None):

    n_components = len(components_to_use)
    signs = np.ones(n_components, dtype=int)
    
    if plot:
        fig, ax = plt.subplots(n_components, 1, figsize=(8, 4*n_components))
    
    for i in range(n_components):
        comp_idx = components_to_use[i]
        spl = CubicSpline(freqs, components[comp_idx, :])
        xnew = np.linspace(band[0], band[1], 100)
        ynew = spl(xnew)
        peaks, _ = find_peaks(ynew)
        valleys, _ = find_peaks(-ynew)
        peaks_prom = peak_prominences(ynew, peaks)[0]
        valley_prom = peak_prominences(-ynew, valleys)[0]

        if np.max(peaks_prom) < np.max(valley_prom):
            signs[i] = -1
            chosen_peak = valleys[np.argmax(valley_prom)]
        else:
            signs[i] = 1
            chosen_peak = peaks[np.argmax(peaks_prom)]

        if plot:
            ax[i].plot(freqs, components[comp_idx, :], '.k', label="data")
            ax[i].plot(xnew, ynew, '-b', label="spline")
            ax[i].scatter(xnew[peaks], ynew[peaks], color='g')
            ax[i].scatter(xnew[valleys], ynew[valleys], color='r')
            ax[i].scatter(xnew[chosen_peak], ynew[chosen_peak], color='m', s=100, label="chosen peak")
            ax[i].set_title(f'PC{comp_idx+1}')
            ax[i].legend() 
            plt.savefig(save_path) if save_path is not None else None
           
    return signs

def reconstruct_signal_from_pca(pca: PCA, pca_result: np.ndarray, components_to_use: list = None) -> np.ndarray:
    """
    Reconstruct the signal from specific PCA components.

    Parameters:
    - pca: The fitted PCA object from scikit-learn.
    - pca_result: The result of the PCA transformation (scores).
    - components_to_use: A list of indices of the components to use for reconstruction.
                         If None, all components are used.

    Returns:
    - The reconstructed signal (powers).
    """
    if components_to_use is not None:
        # Create a copy of the PCA results and zero out the components we don't want to use
        temp_pca_result = np.zeros_like(pca_result)
        temp_pca_result[:, components_to_use] = pca_result[:, components_to_use]
        reconstructed_data = pca.inverse_transform(temp_pca_result)
    else:
        # Use all components for reconstruction
        reconstructed_data = pca.inverse_transform(pca_result)

    # The data was transposed before PCA, so we transpose it back
    return reconstructed_data.T