"""Automatic threshold finding by fitting a Gaussian mixture to a feature."""

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def find_emg_threshold(feature_emg,
                       k_max=4,
                       min_comp_weight=0.01,
                       n_x=1000,
                       random_state=0):
    """
    Return a threshold (in original units) separating two meta-clusters (e.g. wake vs sleep).
    feature_emg: 1D array of positive EMG power samples (no NaNs)
    """
    # 1) sanitize
    feature_emg = np.asarray(feature_emg)
    if feature_emg.size == 0:
        raise ValueError("Empty input")
    if np.any(feature_emg < 0):
        feature_emg = np.clip(feature_emg, a_min=0, a_max=None)

    ft = feature_emg.copy()
    inv_fn = lambda z: z

    # 2) select K by BIC
    best_bic = np.inf
    best_gmm = None
    for k in range(1, min(k_max, len(ft)) + 1):
        g = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
        g.fit(ft.reshape(-1, 1))
        bic = g.bic(ft.reshape(-1, 1))
        if bic < best_bic:
            best_bic = bic
            best_gmm = g

    gmm = best_gmm
    comp_weights = gmm.weights_
    comp_means = gmm.means_.ravel()
    comp_covs = gmm.covariances_.ravel()  # variances since 1D

    # 3) drop/tie very small components: merge into nearest mean
    large_idx = comp_weights >= min_comp_weight
    if not np.any(large_idx):
        # all tiny -> keep the largest component only
        large_idx[np.argmax(comp_weights)] = True

    # Map component index -> active index (for clustering later)
    active_means = comp_means[large_idx]
    active_weights = comp_weights[large_idx]
    active_vars = comp_covs[large_idx]

    # If after removing tiny components there's only one active -> fallback
    if active_means.size == 1:
        # fallback threshold: mean + 0.75*std or 90th percentile (in original units)
        thr = np.percentile(feature_emg, 80)
        return float(thr)

    # 5) cluster components into two meta-clusters (use KMeans on component means)
    km = KMeans(n_clusters=2, random_state=random_state).fit(active_means.reshape(-1, 1))
    labels = km.labels_

    # prepare x grid in transformed space
    x_lo = np.percentile(ft, 0.5)
    x_hi = np.percentile(ft, 99.5)
    x = np.linspace(x_lo, x_hi, n_x)

    # density per GMM component: w_i * N(x | mu_i, sigma_i)
    from scipy.stats import norm
    comp_densities = np.zeros((active_means.size, x.size))
    for i, (w, mu, var) in enumerate(zip(active_weights, active_means, active_vars)):
        sigma = np.sqrt(var)
        comp_densities[i, :] = w * norm.pdf(x, loc=mu, scale=sigma)

    # combined densities for the two meta-clusters
    density_a = comp_densities[labels == 0].sum(axis=0)
    density_b = comp_densities[labels == 1].sum(axis=0)

    # try to find intersection(s) or minima between the combined densities
    diff = density_a - density_b

    # find sign changes (crossings) — candidate thresholds
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    candidate_x = []
    for idx in sign_changes:
        # refine by choosing the point near crossing where total density is minimal
        window = slice(max(0, idx-3), min(x.size, idx+4))
        local_idx = np.argmin((density_a + density_b)[window]) + window.start
        candidate_x.append(x[local_idx])

    # If no sign change, look for local minima in total density between the two highest peaks
    if len(candidate_x) == 0:
        total_pdf = density_a + density_b
        peaks, _ = find_peaks(total_pdf)
        if peaks.size >= 2:
            # find the highest two peaks and search interval between them for minimum
            peak_heights = total_pdf[peaks]
            top2 = peaks[np.argsort(peak_heights)[-2:]]
            lo, hi = sorted(top2)
            min_idx = np.argmin(total_pdf[lo:hi+1]) + lo
            candidate_x.append(x[min_idx])
        else:
            # fallback: percentile in original units
            thr = np.percentile(feature_emg, 80)
            return float(thr)

    # Pick the best candidate: choose one with smallest total density (clean separation)
    candidate_x = np.array(candidate_x)
    total_pdf = (density_a + density_b)
    cand_idxs = [np.argmin(np.abs(x - cx)) for cx in candidate_x]
    chosen_idx = cand_idxs[np.argmin(total_pdf[cand_idxs])]
    threshold_transformed = x[chosen_idx]

    # convert back to original units
    threshold_original = inv_fn(threshold_transformed)
    return float(threshold_original)


def find_pca_threshold(feature_pca,
                       k_max=4,
                       min_comp_weight=0.01,
                       n_x=1000,
                       random_state=0):
    """
    Return a threshold separating two meta-clusters for PCA data.
    This is an adaptation of find_emg_threshold for data that can be negative.
    feature_pca: 1D array of PCA component scores (no NaNs)
    """
    # 1) sanitize
    feature_pca = np.asarray(feature_pca)
    if feature_pca.size == 0:
        raise ValueError("Empty input")

    ft = feature_pca.copy()
    inv_fn = lambda z: z

    # 2) select K by BIC
    best_bic = np.inf
    best_gmm = None
    
    # Ensure k is not greater than number of samples
    max_k = min(k_max, len(np.unique(ft)))
    if max_k == 0:
        return float(np.mean(feature_pca))

    for k in range(1, max_k + 1):
        g = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
        g.fit(ft.reshape(-1, 1))
        bic = g.bic(ft.reshape(-1, 1))
        if bic < best_bic:
            best_bic = bic
            best_gmm = g

    if best_gmm is None:
        return float(np.mean(feature_pca))

    gmm = best_gmm
    comp_weights = gmm.weights_
    comp_means = gmm.means_.ravel()
    comp_covs = gmm.covariances_.ravel()

    # 3) drop very small components
    large_idx = comp_weights >= min_comp_weight
    if not np.any(large_idx):
        large_idx[np.argmax(comp_weights)] = True

    active_means = comp_means[large_idx]
    active_weights = comp_weights[large_idx]
    active_vars = comp_covs[large_idx]

    # If only one component remains, fallback to the mean
    if active_means.size <= 1:
        return float(np.mean(feature_pca))

    # 5) cluster components into two meta-clusters
    km = KMeans(n_clusters=2, random_state=random_state, n_init='auto').fit(active_means.reshape(-1, 1))
    labels = km.labels_

    x_lo, x_hi = np.percentile(ft, [0.5, 99.5])
    x = np.linspace(x_lo, x_hi, n_x)

    from scipy.stats import norm
    comp_densities = np.array([w * norm.pdf(x, loc=mu, scale=np.sqrt(var))
                               for w, mu, var in zip(active_weights, active_means, active_vars)])

    density_a = comp_densities[labels == 0].sum(axis=0)
    density_b = comp_densities[labels == 1].sum(axis=0)

    diff = density_a - density_b
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    candidate_x = []
    if len(sign_changes) > 0:
        for idx in sign_changes:
            window = slice(max(0, idx - 3), min(x.size, idx + 4))
            local_idx = np.argmin((density_a + density_b)[window]) + window.start
            candidate_x.append(x[local_idx])
    
    if len(candidate_x) == 0:
        total_pdf = density_a + density_b
        peaks, _ = find_peaks(total_pdf)
        if peaks.size >= 2:
            peak_heights = total_pdf[peaks]
            top2 = peaks[np.argsort(peak_heights)[-2:]]
            lo, hi = sorted(top2)
            if lo < hi:
                min_idx = np.argmin(total_pdf[lo:hi+1]) + lo
                candidate_x.append(x[min_idx])
        
    if len(candidate_x) == 0:
        return float(np.mean(feature_pca))

    candidate_x = np.array(candidate_x)
    total_pdf_at_candidates = np.interp(candidate_x, x, density_a + density_b)
    threshold_transformed = candidate_x[np.argmin(total_pdf_at_candidates)]
    
    return float(inv_fn(threshold_transformed))