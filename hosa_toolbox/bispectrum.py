# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:37:15 2025

@author: Giannis
"""

# bispectrum.py
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def bispecd(y, nfft=128, wind=5, nsamp=None, overlap=50):
    """
    Direct (FFT-based) bispectrum estimation with frequency-domain smoothing.

    Parameters
    ----------
    y : ndarray
        Input 1D data vector.
    nfft : int
        FFT length [default=128].
    wind : int, 1D array, or 2D array
        Window for smoothing:
        - int: size of Rao-Gabr optimal window
        - 1D array: generates w(i)w(j)w(i+j) style 2D window
        - 2D array: used directly
    nsamp : int
        Samples per segment. If None, auto-select for 8 segments.
    overlap : float
        Percentage overlap [default = 50].

    Returns
    -------
    Bspec : ndarray
        Estimated bispectrum (nfft x nfft), centered (fftshifted).
    waxis : ndarray
        Frequency axis (normalized from -0.5 to 0.5).
    """
    y = y.flatten()
    N = len(y)

    if nsamp is None:
        nsamp = int(N / (8 - 7 * overlap / 100))
    if nfft < nsamp:
        nfft = 2 ** int(np.ceil(np.log2(nsamp)))

    overlap = int(overlap / 100 * nsamp)
    nadvance = nsamp - overlap
    nrecs = (N - overlap) // nadvance

    # ----- Create 2D window -----
    if isinstance(wind, int):
        winsize = wind
        winsize = winsize - (winsize % 2) + 1  # make odd
        if winsize > 1:
            mwind = max(1, int(nfft / winsize))
            lby2 = (winsize - 1) // 2
            theta = np.arange(-lby2, lby2 + 1)

            # Rao-Gabr 2D polynomial window
            opwind = 1 - (2 * mwind / nfft)**2 * (
                theta[:, None]**2 + theta[None, :]**2 + theta[:, None]*theta[None, :]
            )
            hex_mask = (np.abs(theta[:, None]) + np.abs(theta[None, :]) +
                        np.abs(theta[:, None] + theta[None, :])) < winsize
            opwind *= hex_mask
            opwind *= (4 * mwind**2) / (7 * np.pi**2)
        else:
            opwind = np.ones((1, 1))
    elif isinstance(wind, np.ndarray) and wind.ndim == 1:
        w = wind.flatten()
        l = len(w)
        windf = np.concatenate((w[l-1:0:-1], w))
        pad = np.zeros(l-1)
        wpad = np.concatenate((w, pad))
        opwind = np.outer(windf, windf) * np.hankel(w[::-1], w)
    elif isinstance(wind, np.ndarray) and wind.ndim == 2:
        if wind.shape[0] != wind.shape[1]:
            raise ValueError("2D window must be square")
        if wind.shape[0] % 2 == 0:
            raise ValueError("2D window must have odd size")
        opwind = wind
        winsize = wind.shape[0]
    else:
        opwind = np.ones((1, 1))

    # ----- Triple product accumulation -----
    Bspec = np.zeros((nfft, nfft), dtype=complex)
    mask = np.add.outer(np.arange(nfft), np.arange(nfft)) % nfft  # for X(f1+f2)
    locseg = np.arange(nsamp)

    for _ in range(nrecs):
        seg = y[locseg]
        seg = seg - np.mean(seg)
        Xf = np.fft.fft(seg, nfft) / nsamp
        CXf = np.conj(Xf)
        Bspec += (Xf[:, None] * Xf[None, :]) * CXf[mask]
        locseg = locseg + nadvance

    Bspec /= nrecs
    Bspec = np.fft.fftshift(Bspec)

    # ----- Frequency-domain smoothing -----
    if opwind.shape[0] > 1:
        L = opwind.shape[0]
        lby2 = (L - 1) // 2
        Bspec = convolve2d(Bspec, opwind, mode='same')

    # ----- Frequency axis -----
    waxis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0))

    return Bspec, waxis

import numpy as np
import matplotlib.pyplot as plt

def plot_bispecd_dual(Bspec, waxis, title="Bispectrum View (Full vs. 1st Quadrant)",
                      log_scale=False, levels=30, cmap='viridis'):
    """
    Dual plot: full bispectrum and zoomed-in 1st quadrant.

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (fftshifted, complex-valued)
        waxis : 1D np.ndarray
            Normalized frequency axis [-0.5, 0.5]
        log_scale : bool
            Show log10(|B|) if True
        levels : int
            Number of contour levels
        cmap : str
            Colormap
    """
    Babs = np.abs(Bspec)
    if log_scale:
        Babs = np.log10(Babs + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Full Bispectrum Plot ---
    ax = axes[0]
    cf = ax.contourf(waxis, waxis, Babs, levels=levels, cmap=cmap)
    ax.set_title("Full Bispectrum")
    ax.set_xlabel("f1 (normalized)")
    ax.set_ylabel("f2 (normalized)")
    ax.grid(True)
    fig.colorbar(cf, ax=ax, label='log10(|B|)' if log_scale else '|B|')

    # --- First Quadrant Zoom-In ---
    ax = axes[1]
    waxis_pos = waxis[waxis >= 0]
    idx_pos = np.where(waxis >= 0)[0]
    B1Q = Babs[np.ix_(idx_pos, idx_pos)]

    cf1q = ax.contourf(waxis_pos, waxis_pos, B1Q, levels=levels, cmap=cmap)
    ax.set_title("1st Quadrant: $w_1 + w_2 \\leq 0.5$")
    ax.set_xlabel("f1 (normalized)")
    ax.set_ylabel("f2 (normalized)")
    ax.grid(True)

    # Overlay domain boundary: w1 + w2 = 0.5
    f1 = waxis_pos
    f2 = 0.5 - f1
    f2 = np.clip(f2, 0, 0.5)
    mask = (f2 >= 0)
    ax.plot(f1[mask], f2[mask], 'w--', linewidth=2, label='w1 + w2 = 0.5')
    ax.legend()

    fig.colorbar(cf1q, ax=ax, label='log10(|B|)' if log_scale else '|B|')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

