"""Spectral features: EMG band power and PCA of the EEG spectrogram."""


import numpy as np
import scipy
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from nyx.pca import run_pca


def _generate_wavelet_fourier(len_wavelet, f_start, f_stop, deltafreq, sample_rate, f0, normalisation):
    """
    Compute the wavelet coefficients at all scales and compute its Fourier transform.

    Parameters
    ----------
    len_wavelet : int
        length in samples of the wavelet window
    f_start: float
        First frequency in Hz
    f_stop: float
        Last frequency in Hz
    deltafreq : float
        Frequency interval in Hz
    sample_rate : float
        Sample rate in Hz
    f0 : float
    normalisation : float

    Returns:
    -------

    wf : array
        Fourier transform of the wavelet coefficients (after weighting).
        Axis 0 is time; axis 1 is frequency.
        
    Code from Ephyviewer implementation of TimeFrequency Viewer:
    https://github.com/NeuralEnsemble/ephyviewer/blob/master/ephyviewer/timefreqviewer.py
    """
    # compute final map scales
    scales = f0/np.arange(f_start,f_stop,deltafreq)*sample_rate

    # compute wavelet coeffs at all scales
    xi=np.arange(-len_wavelet/2.,len_wavelet/2.)
    xsd = xi[:,np.newaxis] / scales
    wavelet_coefs=np.exp(1j*2.*np.pi*f0*xsd)*np.exp(-np.power(xsd,2)/2.)

    # Normalization
    weighting_function = lambda x: x**(-(1.0+normalisation))
    weighted_wavelet_coefs = wavelet_coefs*weighting_function(scales[np.newaxis,:])

    # Transform the wavelet into the Fourier domain
    wf=scipy.fftpack.fft(weighted_wavelet_coefs,axis=0)
    wf=wf.conj()

    return wf

def _compute_scalogram_ephyviewer(data, min_freq, max_freq, freq_resolution, fs, f0 = 1, exp_corr = 0, time_smooth = 0.5, wanted_size=3.):
    # Code from Ephyviewer implementation of TimeFrequency Viewer:
    # https://github.com/NeuralEnsemble/ephyviewer/blob/master/ephyviewer/timefreqviewer.py

    n_samples = len(data)
    len_wavelet = l = int(2**np.ceil(np.log(wanted_size*fs)/np.log(2)))    
    sig_chunk_size = wanted_size*fs
    downsample_ratio = int(np.ceil(sig_chunk_size/l))
    
    sig_chunk_size = downsample_ratio*l
    sub_sample_rate = fs/downsample_ratio
    
    wavelet_fourrier = _generate_wavelet_fourier(len_wavelet, min_freq, max_freq,
                            freq_resolution, sub_sample_rate, f0, exp_corr)
    
    if downsample_ratio >1:
        n = 8
        q = downsample_ratio
        filter_sos = scipy.signal.cheby1(n, 0.05, 0.8 / q, output='sos')
        
    i_start = 0

    if downsample_ratio>1:
        i_start = i_start - (i_start%downsample_ratio)

    #clip it
    i_start = max(0, i_start)
    i_start = min(i_start, n_samples)
    if downsample_ratio>1:
        #after clip
        i_start = i_start - (i_start%downsample_ratio)

    i_stop = i_start + sig_chunk_size
    i_stop = min(i_stop, n_samples)

    sigs_chunk = data[i_start:i_stop]
    if sigs_chunk.dtype!='float32':
        sigs_chunk = sigs_chunk.astype('float32')

    if downsample_ratio>1:
        small_sig = scipy.signal.sosfiltfilt(filter_sos, sigs_chunk)
        small_sig =small_sig[::downsample_ratio].copy()  # to ensure continuity
    else:
        small_sig = sigs_chunk.copy()# to ensure continuity
    
    left_pad = 0
    if small_sig.shape[0] != wavelet_fourrier.shape[0]:
        #Pad it
        z = np.zeros(wavelet_fourrier.shape[0], dtype=small_sig.dtype)
        left_pad = wavelet_fourrier.shape[0] - small_sig.shape[0]
        z[:small_sig.shape[0]] = small_sig
        small_sig = z
        
    #avoid border effect
    small_sig -= small_sig.mean()

    small_sig_f = scipy.fftpack.fft(small_sig)
    if small_sig_f.shape[0] != wavelet_fourrier.shape[0]:
        print('oulala', small_sig_f.shape, wavelet_fourrier.shape)
        
    wt_tmp=scipy.fftpack.ifft(small_sig_f[:,np.newaxis]*wavelet_fourrier,axis=0)
    wt = scipy.fftpack.fftshift(wt_tmp,axes=[0])
    wt = np.abs(wt).astype('float32')

    # Convert to dB
    wt = 10*np.log10(wt + np.finfo(float).eps)
    
    if left_pad>0:
        wt = wt[:-left_pad]
    
    # Smooth in time
    if time_smooth > 0.:
        n_times = wt.shape[0]
        if n_times > 1:
            # sigma for gaussian filter in units of array indices
            sigma_in_indices = time_smooth * sub_sample_rate
            
            # Create Gaussian kernel
            x = np.arange(n_times)
            kernel = np.exp(-0.5 * ((x - n_times // 2) / sigma_in_indices)**2)
            kernel /= kernel.sum()
            
            # Use rfft for real-valued signal, which is faster
            wt_f = scipy.fft.rfft(wt, axis=0)
            kernel_f = scipy.fft.rfft(np.fft.ifftshift(kernel), n=n_times)
            
            # Convolution in frequency domain is multiplication
            wt_smooth_f = wt_f * kernel_f[:, np.newaxis]
            
            # Inverse FFT to get smoothed signal
            wt = scipy.fft.irfft(wt_smooth_f, n=n_times, axis=0)
        
    return wt.T # shape (n_freq, n_samples)

def _compute_spectrogram_ephyviewer(data: np.ndarray, fs: float, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram in the same way as ephyviewer.spectrogramviewer.
    """
    binsize = params['binsize']
    overlapratio = params['overlapratio']
    scaling = params['scaling']
    detrend = params['detrend']
    mode = params['mode']
    scale = params['scale']
    smoothing = params['time_smooth']
    normalized = params['normalized']

    nperseg = int(binsize * fs)
    noverlap = int(overlapratio * nperseg)

    if noverlap >= nperseg:
        noverlap = nperseg - 1

    if nperseg == 0 or len(data) < nperseg:
        raise ValueError("Data length is too short for the given binsize.")

    freqs, times, Sxx = scipy.signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
                detrend=detrend, scaling=scaling, mode=mode)

    if scale == 'dB':
        if mode == 'psd':
            Sxx = 10. * np.log10(Sxx + np.finfo(float).eps)

    if Sxx is not None:
        Sxx = normalize_spectrogram(Sxx, normalized)

    if smoothing > 0. :
        # Smooth in time using FFT-based convolution for speed
        n_times = Sxx.shape[1]
        if n_times > 1:
            # time_step is the time difference between consecutive columns in Sxx
            time_step = times[1] - times[0]
            # sigma for gaussian filter in units of array indices
            sigma_in_indices = smoothing / time_step
            
            # Create Gaussian kernel
            x = np.arange(n_times)
            kernel = np.exp(-0.5 * ((x - n_times // 2) / sigma_in_indices)**2)
            kernel /= kernel.sum()
            
            # Use rfft for real-valued signal, which is faster
            Sxx_f = scipy.fft.rfft(Sxx, axis=1)
            kernel_f = scipy.fft.rfft(np.fft.ifftshift(kernel), n=n_times)
            
            # Convolution in frequency domain is multiplication
            Sxx_smooth_f = Sxx_f * kernel_f[np.newaxis, :]
            
            # Inverse FFT to get smoothed signal
            Sxx = scipy.fft.irfft(Sxx_smooth_f, n=n_times, axis=1)

    return Sxx, freqs, times


def normalize_scalogram(scalogram: np.ndarray) -> np.ndarray:
    """Normalize the scalogram by subtracting its mean power."""
    mean_power = np.mean(scalogram)
    return scalogram - mean_power


def normalize_spectrogram(Sxx: np.ndarray, mode) -> np.ndarray:
    """Normalise a spectrogram of shape ``(n_freqs, n_times)``.

    ``mode`` is one of:

    ``False`` / ``None``
        No normalisation.
    ``True`` / ``"mean"``
        Subtract the global mean power. What the rodent parameter files use.
    ``"zscore"``
        Z-score each frequency bin across time (row-wise). Removes absolute
        power differences, so PCA is driven by the *pattern* of spectral change
        rather than by amplitude -- which is what makes components comparable
        across subjects. What the human parameter files use.
    ``"relative"``
        Scale each time bin so its spectrum sums to 1 (column-wise). Also
        removes global amplitude, but preserves spectral shape; not equivalent
        to z-scoring.
    """
    if mode is False or mode is None:
        return Sxx
    if mode is True or mode == "mean":
        return Sxx - Sxx.mean()
    if mode == "zscore":
        mean = Sxx.mean(axis=1, keepdims=True)
        std = Sxx.std(axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)  # flat bins would divide by zero
        return (Sxx - mean) / std
    if mode == "relative":
        totals = Sxx.sum(axis=0, keepdims=True)
        totals = np.where(totals == 0, 1.0, totals)  # silent epochs
        return Sxx / totals
    raise ValueError(
        f"Unknown 'normalized' setting {mode!r}. "
        f"Expected false, 'mean', 'zscore' or 'relative'."
    )

def band_power(freqs, power_smooth, f_band):
    idx_band = np.where((freqs >= f_band[0]) & (freqs <= f_band[1]))[0]
    return np.sum(power_smooth[idx_band, :], axis=0)


def compute_emg_power_trace(emg: np.ndarray, fs_signal: float, emg_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute EMG spectrogram and collapse to a band-power 1D trace with its own 'fs'.
    Returns (power_trace, fs_trace).
    """
    Sxx, freqs, times = _compute_spectrogram_ephyviewer(
        data=emg,
        fs=fs_signal,
        params=emg_params,
    )
    if Sxx is None or times is None or len(times) < 2:
        raise RuntimeError("EMG spectrogram failed or too short.")

    # Sum across the EMG band of interest (here: using [min_freq, max_freq] from params)
    f_band = (float(emg_params["min_freq"]), float(emg_params["max_freq"]))
    power = band_power(freqs=freqs, power_smooth=Sxx, f_band=f_band)

    # The time vector spacing sets the effective sampling rate for the power trace
    dt = float(times[1] - times[0])
    fs_power = 1.0 / dt if dt > 0 else 1.0

    # Replace inf/nan with a sentinel outside the valid range
    power = np.asarray(power, dtype=float)
    invalid = ~np.isfinite(power)
    if invalid.any():
        finite_vals = power[np.isfinite(power)]
        if finite_vals.size:
            mn = finite_vals.min()
            mx = finite_vals.max()
            delta = mx - mn if np.isfinite(mx - mn) and (mx - mn) > 0 else 1.0
            sentinel = mn - max(1.0, 0.1 * delta)  # put it below the minimum by a margin
        else:
            sentinel = -np.finfo(power.dtype).max  # extreme negative if nothing is finite
        power[invalid] = sentinel
        
    smoothing_window_emg = int(emg_params['time_smooth'] * fs_power)
    power_smoothed = uniform_filter1d(power, size=smoothing_window_emg)
    power_smoothed = MinMaxScaler().fit_transform(power_smoothed.reshape(-1, 1)).flatten().reshape(power_smoothed.shape)
   
    return Sxx, freqs, times, power_smoothed, fs_power

def compute_eeg_pca_feature(eeg: np.ndarray, fs_signal: float, params: dict, wakesleep_hypno: dict, within: str = "SLEEP") -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA, np.ndarray, np.ndarray, float]:
    """
    Compute the EEG spectrogram and run PCA on the epochs currently labelled
    ``within``, returning the component scores aligned to the spectrogram time
    vector plus its sampling rate.

    Fitting inside one label is what lets the pipeline run several clustering
    steps: the first step fits over everything (or over SLEEP), and a later step
    refits inside WAKE or inside SLEEP so its components describe only the
    structure that is left to resolve.
    """
    Sxx, freqs, times = _compute_spectrogram_ephyviewer(
        data=eeg,
        fs=fs_signal,
        params=params["EEG"]
    )
    if Sxx is None or times is None or len(times) < 2:
        raise RuntimeError("EEG spectrogram failed or too short.")

    dt = float(times[1] - times[0])
    fs_bins = 1.0 / dt if dt > 0 else 1.0
    
    
    idx_band_ecog = np.where((freqs >= params["EEG"]['min_freq']) & (freqs <= params["EEG"]['max_freq']))[0]
    Sxx_ecog_cut = Sxx[idx_band_ecog, :]
    freqs_ecog_cut = freqs[idx_band_ecog]

    pca, pca_result, pca_signal = run_pca(Sxx_ecog_cut, fs_bins, wakesleep_hypno=wakesleep_hypno, label=within, n_components=params['scoring']['pc_components'])
    if pca_result.shape[0] == 0:
        raise ValueError(
            f"No epochs labelled {within!r} were found, so there is nothing to run PCA on. "
            f"Available labels: {sorted(set(np.asarray(wakesleep_hypno['label']).tolist()))}."
        )

    return Sxx_ecog_cut, freqs_ecog_cut, times, pca, pca_result, pca_signal, fs_bins
