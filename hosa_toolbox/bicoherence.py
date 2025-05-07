# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:46:07 2025

@author: Giannis
"""

# bispectrum.py (add this function)
import numpy as np
from scipy.signal import get_window
from scipy.fft import fft, fftshift

def bicoher(y, nfft=128, wind=None, nsamp=None, overlap=50):
    """
    Estimate bicoherence using the direct FFT-based method.

    Parameters
    ----------
    y : ndarray
        1D input data (real or complex).
    nfft : int
        FFT length [default = 128].
    wind : str or ndarray
        Window function name (e.g., 'hanning') or custom array of length nsamp.
    nsamp : int
        Samples per segment. If None, will auto-determine for ~8 segments.
    overlap : float
        Overlap percentage [default = 50].

    Returns
    -------
    bic : ndarray
        Bicoherence matrix (nfft x nfft), values in [0, 1], centered.
    waxis : ndarray
        Normalized frequency axis (fftshifted).
    """
    y = y.flatten()
    N = len(y)

    if nsamp is None:
        nsamp = int(N / (8 - 7 * overlap / 100))
    if nfft < nsamp:
        nfft = 2 ** int(np.ceil(np.log2(nsamp)))

    overlap = int(overlap / 100 * nsamp)
    step = nsamp - overlap
    nrecs = (N - overlap) // step

    # Create window
    if wind is None:
        wind = get_window("hann", nsamp)
    elif isinstance(wind, str):
        wind = get_window(wind, nsamp)
    else:
        wind = np.asarray(wind)
        if wind.shape[0] != nsamp:
            print("Window length mismatch. Using default Hanning window.")
            wind = get_window("hann", nsamp)

    wind = wind.flatten()
    bic = np.zeros((nfft, nfft), dtype=np.complex128)
    Pyy = np.zeros(nfft)
    mask = np.add.outer(np.arange(nfft), np.arange(nfft)) % nfft
    Yf12 = np.zeros((nfft, nfft), dtype=np.complex128)

    for k in range(nrecs):
        start = k * step
        end = start + nsamp
        if end > N:
            break

        seg = y[start:end]
        seg = (seg - np.mean(seg)) * wind
        Yf = fft(seg, nfft) / nsamp
        CYf = np.conj(Yf)

        Pyy += np.abs(Yf)**2
        Yf12[:, :] = CYf[mask]
        bic += (Yf[:, None] * Yf[None, :]) * Yf12

    # Normalize
    bic /= nrecs
    Pyy /= nrecs
    norm = (Pyy[:, None] * Pyy[None, :]) * Pyy[mask]
    bic = np.abs(bic)**2 / norm
    bic = fftshift(bic)

    # Frequency axis
    waxis = np.fft.fftshift(np.fft.fftfreq(nfft))

    return bic, waxis

def plot_bicoher(bic, waxis):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.contourf(waxis, waxis, bic, levels=10, cmap='plasma')
    plt.colorbar(label='Bicoherence')
    plt.xlabel("f1 (normalized)")
    plt.ylabel("f2 (normalized)")
    plt.title("Bicoherence via Direct Method")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Display max bicoherence value
    max_val = np.max(bic)
    max_idx = np.unravel_index(np.argmax(bic), bic.shape)
    print(f"Max bicoherence at (f1={waxis[max_idx[0]]:.3f}, f2={waxis[max_idx[1]]:.3f}) = {max_val:.3f}")
