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

def plot_bispecd(Bspec, waxis, title="Bispectrum via Direct Method with Smoothing",
                 log_scale=False, levels=30, cmap="viridis", save_path=None,
                 show_domain=True):
    """
    Enhanced bispectrum plot with optional log scale, contour level control,
    domain boundary overlay, and save option.

    Parameters:
        Bspec : 2D np.ndarray
            Bispectrum matrix (fftshifted, complex-valued)
        waxis : 1D np.ndarray
            Normalized frequency axis [-0.5, 0.5]
        title : str
            Plot title
        log_scale : bool
            If True, show log10(|Bspec|) instead of linear magnitude
        levels : int
            Number of contour levels
        cmap : str
            Colormap name
        save_path : str or None
            If provided, saves the figure to this file path
        show_domain : bool
            If True, overlays the w1 + w2 ≤ 0.5 domain boundary
    """
    Babs = np.abs(Bspec)
    if log_scale:
        Babs = np.log10(Babs + 1e-10)

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(waxis, waxis, Babs, levels=levels, cmap=cmap)
    plt.colorbar(contour, label='log10(|B(f1,f2)|)' if log_scale else '|B(f1,f2)|')

    plt.xlabel("f1 (normalized)")
    plt.ylabel("f2 (normalized)")
    plt.title(title)
    plt.grid(True)

    if show_domain:
        # Overlay domain boundary w1 + w2 = 0.5 in first quadrant
        w_positive = waxis[waxis >= 0]
        f1_domain = w_positive
        f2_domain = 0.5 - f1_domain
        f2_domain = np.clip(f2_domain, 0, 0.5)
        plt.plot(f1_domain, f2_domain, 'w--', linewidth=2, label='w1 + w2 = 0.5')
        plt.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Plot saved to {save_path}")
    plt.show()

