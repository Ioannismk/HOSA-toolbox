# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:12:33 2025

@author: Giannis
"""

import numpy as np
from scipy.ndimage import label

def bspec_areas(Bspec, waxis):
    """
    Compute bispectral total area and dominant peak area within the 1st quadrant:
    w1, w2 ≥ 0 and w1 + w2 ≤ 0.5 (normalized frequency in rad/2π).

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (assumed centered via fftshift)
        waxis : 1D np.ndarray
            Normalized frequency axis (from -0.5 to +0.5)

    Returns:
        total_area : float
            Area above threshold within the domain of interest
        max_area : float
            Area around dominant peak within radius and domain
    """
    Babs = np.abs(Bspec)
    nfft = Bspec.shape[0]
    zero_idx = nfft // 2 if nfft % 2 == 0 else (nfft - 1) // 2

    # Define the domain: first quadrant and w1 + w2 <= 0.5
    w1, w2 = np.meshgrid(waxis, waxis)
    domain_mask = (w1 >= 0) & (w2 >= 0) & ((w1 + w2) <= 0.5)

    # Restrict to domain for peak detection
    Bdomain = np.copy(Babs)
    Bdomain[~domain_mask] = 0
    peak_idx = np.unravel_index(np.argmax(Bdomain), Bdomain.shape)
    peak_val = Babs[peak_idx]
    peak_coords = np.array([waxis[peak_idx[1]], waxis[peak_idx[0]]])

    # Thresholds
    levels = np.linspace(Babs.min(), peak_val, 15)
    thresh_tot = levels[0]
    thresh_peak = levels[0] + (peak_val - levels[0]) * 2/3

    # Total area mask
    mask_total = (Babs >= thresh_tot) & domain_mask

    # Distance from peak (normalized frequency space)
    w1_flat, w2_flat = w1[domain_mask], w2[domain_mask]
    coords = np.stack([w1_flat, w2_flat], axis=1)
    dists = np.linalg.norm(coords - peak_coords, axis=1)
    dom_radius = np.max(dists[(Babs[domain_mask] >= thresh_peak)])  # estimated radius

    # Max area mask
    mask_peak = domain_mask.copy()
    for i in range(nfft):
        for j in range(nfft):
            if not domain_mask[j, i]:
                mask_peak[j, i] = False
                continue
            dist = np.linalg.norm([waxis[i] - peak_coords[0], waxis[j] - peak_coords[1]])
            if dist > dom_radius:
                mask_peak[j, i] = False

    # Approximate area via pixel size
    pixel_area = (waxis[1] - waxis[0]) ** 2
    total_area = np.sum(mask_total) * pixel_area
    max_area = np.sum(mask_peak & (Babs >= thresh_tot)) * pixel_area

    return total_area, max_area

def bspec_entropy(Bspec):
    """
    Compute the bispectral entropy in the principal region:
    w1, w2 ≥ 0, w2 ≤ w1, and w1 + w2 ≤ 0.5 (normalized units)

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (fftshifted)

    Returns:
        BS_Entropy : float
            Entropy in the principal bispectral domain
    """
    Babs = np.abs(Bspec)
    nfft = Babs.shape[0]
    zero_idx = nfft // 2 if nfft % 2 == 0 else (nfft - 1) // 2
    waxis = np.linspace(-0.5, 0.5, nfft, endpoint=False)

    pmn = []

    for i in range(zero_idx, nfft):
        for j in range(zero_idx, nfft):
            if j > i:
                continue
            if waxis[i] + waxis[j] > 0.5:
                continue
            pmn.append(Babs[j, i])

    pmn = np.array(pmn)
    if pmn.sum() == 0:
        return 0.0  # avoid division by zero

    pmn = pmn / pmn.sum()
    BS_Entropy = -np.sum(pmn * np.log(pmn + 1e-10))  # add epsilon for stability

    return BS_Entropy

def bspec_max_area(Bspec, waxis):
    """
    Compute area under the dominant bispectral peak within the 1st quadrant,
    where w1, w2 ≥ 0 and w1 + w2 ≤ 0.5.

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (fftshifted)
        waxis : 1D np.ndarray
            Normalized frequency axis (from -0.5 to 0.5)

    Returns:
        max_area : float
            Estimated area under dominant peak
    """
    Babs = np.abs(Bspec)
    nfft = Babs.shape[0]
    zero_idx = nfft // 2 if nfft % 2 == 0 else (nfft - 1) // 2

    w1, w2 = np.meshgrid(waxis, waxis)
    domain_mask = (w1 >= 0) & (w2 >= 0) & ((w1 + w2) <= 0.5)

    Bdomain = np.copy(Babs)
    Bdomain[~domain_mask] = 0

    # Locate peak
    peak_idx = np.unravel_index(np.argmax(Bdomain), Bdomain.shape)
    peak_val = Babs[peak_idx]
    peak_coords = np.array([waxis[peak_idx[1]], waxis[peak_idx[0]]])

    if (peak_coords[0] + peak_coords[1]) > 0.5:
        raise ValueError("No dominant peak in the domain of interest.")

    # Contour threshold approximation
    levels = np.linspace(Babs.min(), peak_val, 15)
    threshold = levels[0] + (peak_val - levels[0]) * 2 / 3

    # Estimate radius from peak: points with magnitude above threshold
    coords = np.argwhere((Babs >= threshold) & domain_mask)
    coords_freq = np.array([[waxis[c[1]], waxis[c[0]]] for c in coords])
    dists = np.linalg.norm(coords_freq - peak_coords, axis=1)
    dom_radius = np.max(dists)

    # Compute area under peak
    max_area = 0.0
    pixel_area = (waxis[1] - waxis[0]) ** 2

    for i in range(zero_idx, nfft):
        for j in range(zero_idx, nfft):
            if not (waxis[i] + waxis[j] <= 0.5):
                continue
            dist = np.linalg.norm([waxis[i] - peak_coords[0], waxis[j] - peak_coords[1]])
            if dist > dom_radius:
                continue
            if i == zero_idx and j == zero_idx:
                max_area += pixel_area / 4
            elif waxis[i] == 0.5 or waxis[j] == 0.5:
                max_area += pixel_area / 8
            elif (waxis[i] + waxis[j] == 0.5) or i == zero_idx or j == zero_idx:
                max_area += pixel_area / 2
            else:
                max_area += pixel_area

    return max_area

def bspec_sq_entropy(Bspec):
    """
    Compute the squared bispectral entropy in the principal region:
    w1, w2 ≥ 0, w2 ≤ w1, and w1 + w2 ≤ 0.5 (normalized units).

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (fftshifted)

    Returns:
        BS_sq_Entropy : float
            Squared entropy in the principal bispectral domain
    """
    Babs = np.abs(Bspec)
    Bsq = Babs ** 2
    nfft = Babs.shape[0]
    zero_idx = nfft // 2 if nfft % 2 == 0 else (nfft - 1) // 2
    waxis = np.linspace(-0.5, 0.5, nfft, endpoint=False)

    qmn = []

    for i in range(zero_idx, nfft):
        for j in range(zero_idx, nfft):
            if j > i or (waxis[i] + waxis[j] > 0.5):
                continue
            qmn.append(Bsq[j, i])

    qmn = np.array(qmn)
    if qmn.sum() == 0:
        return 0.0

    qmn = qmn / np.sum(qmn)
    BS_sq_Entropy = -np.sum(qmn * np.log(qmn + 1e-10))  # log stabilizer

    return BS_sq_Entropy

def bspec_total_area(Bspec, waxis):
    """
    Compute the total area of the bispectrum above a threshold in the 1st quadrant:
    where w1, w2 ≥ 0 and w1 + w2 ≤ 0.5 (normalized).

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (fftshifted)
        waxis : 1D np.ndarray
            Normalized frequency axis (from -0.5 to +0.5)

    Returns:
        total_area : float
            Estimated bispectral area in the domain of interest
    """
    Babs = np.abs(Bspec)
    nfft = Babs.shape[0]
    zero_idx = nfft // 2 if nfft % 2 == 0 else (nfft - 1) // 2

    w1, w2 = np.meshgrid(waxis, waxis)
    domain_mask = (w1 >= 0) & (w2 >= 0) & ((w1 + w2) <= 0.5)

    # Threshold from lowest contour level
    levels = np.linspace(Babs.min(), Babs.max(), 15)
    threshold = levels[0]

    pixel_area = (waxis[1] - waxis[0]) ** 2
    total_area = 0.0

    for i in range(zero_idx, nfft):
        for j in range(zero_idx, nfft):
            if not (waxis[i] + waxis[j] <= 0.5):
                continue
            if Babs[j, i] < threshold:
                continue
            if i == zero_idx and j == zero_idx:
                total_area += pixel_area / 4
            elif waxis[i] == 0.5 or waxis[j] == 0.5:
                total_area += pixel_area / 8
            elif (waxis[i] + waxis[j] == 0.5) or i == zero_idx or j == zero_idx:
                total_area += pixel_area / 2
            else:
                total_area += pixel_area

    return total_area

def dominant_peak(Bspec, fs):
    """
    Find the dominant peak of the bispectrum and its corresponding frequencies.

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (fftshifted)
        fs : float
            Sampling frequency in Hz

    Returns:
        peak_value : float
            Maximum bispectrum magnitude in the 1st quadrant
        peak_f1 : float
            Frequency along x-axis (Hz) of dominant peak
        peak_f2 : float
            Frequency along y-axis (Hz) of dominant peak
    """
    Babs = np.abs(Bspec)
    nfft = Babs.shape[0]
    zero_idx = nfft // 2 if nfft % 2 == 0 else (nfft - 1) // 2
    waxis = np.linspace(-0.5, 0.5, nfft, endpoint=False)

    # Limit to 1st quadrant: w1, w2 >= 0 and w1 + w2 <= 0.5
    w1, w2 = np.meshgrid(waxis, waxis)
    domain_mask = (w1 >= 0) & (w2 >= 0) & ((w1 + w2) <= 0.5)
    Bdomain = np.copy(Babs)
    Bdomain[~domain_mask] = 0

    peak_idx = np.unravel_index(np.argmax(Bdomain), Bdomain.shape)
    peak_value = Babs[peak_idx]
    peak_w1 = waxis[peak_idx[1]]
    peak_w2 = waxis[peak_idx[0]]

    if peak_w1 + peak_w2 > 0.5:
        raise ValueError("No dominant peak found inside valid domain.")

    # Convert from normalized frequency to Hz
    peak_f1 = peak_w1 * fs
    peak_f2 = peak_w2 * fs

    return peak_value, peak_f1, peak_f2

def find_dom_peak_radius(Babs, waxis, threshold, zero_idx, dom_peak_row, dom_peak_col):
    """
    Estimate the radius around the dominant peak at which bispectral energy drops below a threshold.

    Parameters:
        Babs : 2D np.ndarray
            Magnitude of the bispectrum matrix
        waxis : 1D np.ndarray
            Normalized frequency axis (from -0.5 to 0.5)
        threshold : float
            Value below which bispectrum is considered dropped
        zero_idx : int
            Index in waxis corresponding to 0 frequency
        dom_peak_row : int
            Row index (y-axis) of dominant peak in Bspec
        dom_peak_col : int
            Column index (x-axis) of dominant peak in Bspec

    Returns:
        dom_peak_radius : float
            Radius in frequency space to closest below-threshold point
    """
    nfft = len(waxis)
    dom_peak_coord = np.array([waxis[dom_peak_col], waxis[dom_peak_row]])
    min_radius = np.inf

    for i in range(zero_idx, nfft):
        for j in range(zero_idx, nfft):
            if waxis[i] + waxis[j] > 0.5:
                continue
            if Babs[j, i] < threshold:
                dist = np.linalg.norm(np.array([waxis[i], waxis[j]]) - dom_peak_coord)
                if dist < min_radius:
                    min_radius = dist

    return min_radius

def compute_bispectral_features(Bspec, waxis, fs):
    """
    Wrapper to compute all bispectral features from Bspec matrix.

    Returns:
        A dictionary with named features.
    """
    total_area, max_area = bspec_areas(Bspec, waxis)
    entropy = bspec_entropy(Bspec)
    sq_entropy = bspec_sq_entropy(Bspec)
    peak_val, peak_f1, peak_f2 = dominant_peak(Bspec, fs)

    return {
        "total_area": total_area,
        "max_area": max_area,
        "entropy": entropy,
        "squared_entropy": sq_entropy,
        "peak_value": peak_val,
        "peak_f1": peak_f1,
        "peak_f2": peak_f2,
    }

