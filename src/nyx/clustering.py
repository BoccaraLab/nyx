"""Clustering sleep epochs in PC space, and mapping clusters back onto time."""

import numpy as np
import pandas as pd


def run_clustering_step(features_scaled: np.ndarray, scaler, clustering_params: dict):
    """
    Run a configurable clustering algorithm on scaled features.

    Parameters
    ----------
    features_scaled : np.ndarray, shape (N, D)
        Standardised feature matrix.
    scaler : fitted StandardScaler
        Used to inverse-transform centres (kept for API consistency; centres
        are returned in the scaled space so they can be used for sorting).
    clustering_params : dict
        method                   : 'kmeans' | 'gmm' | 'hdbscan' | 'elliptic'
        n_clusters               : int  (used by kmeans / gmm)
        hdbscan_min_cluster_size : int
        hdbscan_min_samples      : int
        elliptic_contamination   : float
        elliptic_support_fraction: float

    Returns
    -------
    labels         : np.ndarray[int]  final cluster labels (0 .. n_clusters-1, no -1)
    raw_labels     : np.ndarray[int]  raw labels from algorithm (-1 = noise for density methods)
    cluster_centers: np.ndarray, shape (max_label + 1, D), in scaled space,
                     indexed by cluster id (cluster_centers[cid])
    """
    import hdbscan as hdbscan_lib
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    from sklearn.mixture import GaussianMixture

    method = clustering_params.get('method', 'kmeans')
    n_clusters = int(clustering_params.get('n_clusters', 2))

    def _assign_noise_to_nearest(raw, centers, unique_clusters):
        """Assign noise points (-1) to the nearest cluster centroid."""
        labels = raw.copy()
        noise = raw == -1
        if noise.sum() > 0:
            dists = pairwise_distances(features_scaled[noise], centers)
            labels[noise] = unique_clusters[np.argmin(dists, axis=1)]
        return labels

    def _reindex(labels, unique_clusters):
        """Remap arbitrary cluster IDs to contiguous 0..n-1."""
        lmap = {c: i for i, c in enumerate(unique_clusters)}
        return np.array([lmap[c] for c in labels], dtype=int)

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        raw_labels = model.fit_predict(features_scaled)
        cluster_centers = model.cluster_centers_
        return raw_labels.astype(int), raw_labels.astype(int), cluster_centers

    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=0)
        model.fit(features_scaled)
        raw_labels = model.predict(features_scaled).astype(int)
        cluster_centers = model.means_
        return raw_labels, raw_labels.copy(), cluster_centers

    elif method == 'hdbscan':
        min_cs = int(clustering_params.get('hdbscan_min_cluster_size', 50))
        min_s = int(clustering_params.get('hdbscan_min_samples', 1))
        model = hdbscan_lib.HDBSCAN(min_cluster_size=min_cs, min_samples=min_s)
        raw_labels = model.fit_predict(features_scaled).astype(int)
        unique_clusters = np.unique(raw_labels[raw_labels >= 0])
        if len(unique_clusters) == 0:
            # Fallback when HDBSCAN finds nothing
            fallback = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            fb_labels = fallback.fit_predict(features_scaled).astype(int)
            return fb_labels, raw_labels, fallback.cluster_centers_
        centers = np.array([features_scaled[raw_labels == c].mean(axis=0) for c in unique_clusters])
        labels = _reindex(_assign_noise_to_nearest(raw_labels, centers, unique_clusters), unique_clusters)
        return labels, raw_labels, centers

    elif method == 'elliptic':
        # Fits a robust Gaussian to the bulk of the data and calls the tail
        # outliers. Used for wake/sleep in short recordings, where sleep forms
        # one dense blob and wake is the scattered minority -- so a method that
        # models "the bulk plus everything else" beats one that assumes two
        # comparable clusters.
        from sklearn.covariance import EllipticEnvelope

        contamination = float(clustering_params.get('elliptic_contamination', 0.1))
        support_fraction = clustering_params.get('elliptic_support_fraction', 0.75)
        model = EllipticEnvelope(
            contamination=contamination,
            support_fraction=(None if support_fraction is None
                              else float(support_fraction)),
            random_state=0,
        )
        # EllipticEnvelope returns +1 for inliers and -1 for outliers; remap to
        # 0 = the dense component, 1 = the tail.
        raw_labels = np.where(model.fit_predict(features_scaled) == 1, 0, 1).astype(int)
        unique_clusters = np.unique(raw_labels)
        centers = np.array([features_scaled[raw_labels == c].mean(axis=0)
                            for c in unique_clusters])
        return _reindex(raw_labels, unique_clusters), raw_labels, centers

    else:
        raise ValueError(f"Unknown clustering method: '{method}'. "
                         f"Supported: 'kmeans', 'gmm', 'hdbscan', 'elliptic'.")


def reconstruct_signal_multiclass(stage_array: np.ndarray, wakesleep_hypno: dict,
                                   fs_new_eeg: float, full_length: int,
                                   within: str = 'SLEEP',
                                   outside: int = -1) -> np.ndarray:
    """
    Map a per-epoch stage array, covering only the epochs labelled ``within``,
    back onto the full epoch grid.

    Epochs outside ``within`` are set to ``outside``; NOSIGNAL stretches are
    always preserved as -2. Typical values: -2=NOSIGNAL, -1=WAKE, 0=NREM, 1=REM.

    Parameters
    ----------
    stage_array : np.ndarray[int]
        Integer stage labels, one per ``within`` epoch, in chronological order.
    wakesleep_hypno : dict with 'time', 'duration', 'label'
        The labelling produced so far.
    fs_new_eeg : float  sampling frequency of the epoch grid
    full_length : int   total number of epochs
    within : str
        Which label this step subdivided.
    outside : int
        Code written to epochs this step did not touch.

    Returns
    -------
    cluster_signal : np.ndarray[int], length = full_length
    """
    cluster_signal = np.full(full_length, outside, dtype=int)
    wake_sleep_df = pd.DataFrame(wakesleep_hypno)

    # Preserve NOSIGNAL periods from wakesleep_hypno (mark as -2)
    for _, row in wake_sleep_df.iterrows():
        if row['label'] == 'NOSIGNAL':
            start_idx = int(row['time'] * fs_new_eeg)
            end_idx = int((row['time'] + row['duration']) * fs_new_eeg)
            start_idx = max(0, min(start_idx, full_length))
            end_idx = max(0, min(end_idx, full_length))
            cluster_signal[start_idx:end_idx] = -2

    cluster_pointer = 0
    total = len(stage_array)

    for _, row in wake_sleep_df.iterrows():
        if row['label'] == within:
            # Must use int() (truncation), NOT np.round(), to match run_pca
            start_idx = int(row['time'] * fs_new_eeg)
            end_idx   = int((row['time'] + row['duration']) * fs_new_eeg)
            start_idx = max(0, min(start_idx, full_length))
            end_idx   = max(0, min(end_idx,   full_length))
            segment_len = end_idx - start_idx
            if segment_len <= 0:
                continue
            num_to_take = min(segment_len, total - cluster_pointer)
            if num_to_take <= 0:
                break
            cluster_signal[start_idx:start_idx + num_to_take] = \
                stage_array[cluster_pointer:cluster_pointer + num_to_take]
            cluster_pointer += num_to_take

    return cluster_signal