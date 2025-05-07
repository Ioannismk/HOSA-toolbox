# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:18:54 2025

@author: Giannis
"""

# cumulants.py
import numpy as np

def cum2est(y, maxlag, nsamp, overlap, flag='biased'):
    """
    Estimate the second-order cumulant (autocovariance) of signal y.

    Parameters
    ----------
    y : np.ndarray
        Input 1D signal.
    maxlag : int
        Maximum lag to compute.
    nsamp : int
        Segment length. If <= 0, the function processes the full signal as one segment.
    overlap : float
        Percentage of overlap between segments (0-100).
    flag : str
        'biased' or 'unbiased' estimation.

    Returns
    -------
    y_cum : np.ndarray
        Estimated second-order cumulant for lags -maxlag to maxlag.
    """
    y = y.flatten()
    N = len(y)

    if nsamp <= 0:
        nsamp = N
    overlap = int(overlap / 100 * nsamp)
    nadvance = nsamp - overlap
    nrecord = (N - overlap) // nadvance

    y_cum = np.zeros(maxlag + 1, dtype=np.complex128)

    for i in range(nrecord):
        start = i * nadvance
        end = start + nsamp
        if end > N:
            break
        x = y[start:end]
        x = x - np.mean(x)
        for k in range(maxlag + 1):
            y_cum[k] += np.dot(np.conj(x[:nsamp - k]), x[k:])

    if flag.lower().startswith('b'):
        y_cum /= (nsamp * nrecord)
    else:
        y_cum /= (nrecord * (nsamp - np.arange(maxlag + 1)))

    if maxlag > 0:
        y_cum = np.concatenate((np.conj(y_cum[1:][::-1]), y_cum))

    return y_cum

def cum2x(x, y, maxlag, nsamp, overlap, flag='biased'):
    """
    Estimate cross-covariance between x and y:
    Cxy(m) = E[x*(n) * y(n + m)]

    Parameters
    ----------
    x, y : np.ndarray
        Input signals (1D, real or complex). Must have the same shape.
    maxlag : int
        Maximum lag to compute (-maxlag to maxlag).
    nsamp : int
        Segment length.
    overlap : float
        Overlap percentage (0-100).
    flag : str
        'biased' or 'unbiased'.

    Returns
    -------
    y_cum : np.ndarray
        Cross-covariance estimates from -maxlag to +maxlag.
    """
    x = x.flatten()
    y = y.flatten()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    N = len(x)
    overlap = int(overlap / 100 * nsamp)
    nadvance = nsamp - overlap
    nrecord = (N - overlap) // nadvance

    nlags = 2 * maxlag + 1
    zlag = maxlag
    y_cum = np.zeros(nlags, dtype=np.complex128)

    # Scaling
    if flag.lower().startswith('b'):
        scale = np.ones(nlags) / nsamp
    else:
        lags = np.arange(-maxlag, maxlag + 1)
        scale = 1.0 / (nsamp - np.abs(lags))

    for i in range(nrecord):
        start = i * nadvance
        end = start + nsamp
        if end > N:
            break
        xs = x[start:end] - np.mean(x[start:end])
        ys = y[start:end] - np.mean(y[start:end])

        y_cum[zlag] += np.dot(np.conj(xs), ys)
        for m in range(1, maxlag + 1):
            y_cum[zlag - m] += np.dot(np.conj(xs[m:]), ys[:nsamp - m])
            y_cum[zlag + m] += np.dot(np.conj(xs[:nsamp - m]), ys[m:])

    y_cum *= scale / nrecord
    return y_cum

def cum3est(y, maxlag, nsamp, overlap, flag='biased', k1=0):
    """
    Estimate third-order cumulants: C3(m, k1) = E[x*(n) * x(n+m) * x(n+k1)]

    Parameters
    ----------
    y : np.ndarray
        Input 1D signal (real or complex).
    maxlag : int
        Maximum lag m to compute (-maxlag to +maxlag).
    nsamp : int
        Samples per segment (used for averaging).
    overlap : float
        Percentage overlap (0-100).
    flag : str
        'biased' or 'unbiased' estimate.
    k1 : int
        Fixed lag value in C3(m, k1).

    Returns
    -------
    y_cum : np.ndarray
        Estimated third-order cumulant array of length (2*maxlag + 1).
        Indexed as y_cum[m + maxlag]
    """
    y = y.flatten()
    N = len(y)
    minlag = -maxlag
    overlap = int(overlap / 100 * nsamp)
    nadvance = nsamp - overlap
    nrecord = (N - overlap) // nadvance

    nlags = 2 * maxlag + 1
    zlag = maxlag  # center index for zero lag

    y_cum = np.zeros(nlags, dtype=np.complex128)

    # Scaling vector for bias correction
    if flag.lower().startswith('b'):
        scale = np.ones(nlags) / nsamp
    else:
        lsamp = nsamp - abs(k1)
        scale = np.array([1.0 / (lsamp - abs(lag)) for lag in range(-maxlag, maxlag + 1)])

    for i in range(nrecord):
        start = i * nadvance
        end = start + nsamp
        if end > N:
            break

        x = y[start:end]
        x = x - np.mean(x)
        cx = np.conj(x)
        z = np.zeros_like(x)

        # Compute x(n)*conj(x(n+k1)) term
        if k1 >= 0:
            z[:nsamp - k1] = x[:nsamp - k1] * cx[k1:]
        else:
            z[-k1:] = x[-k1:] * cx[:nsamp + k1]

        # C3(0, k1)
        y_cum[zlag] += np.dot(z, x)

        # C3(m ≠ 0, k1)
        for k in range(1, maxlag + 1):
            y_cum[zlag - k] += np.dot(z[k:], x[:nsamp - k])   # m = -k
            y_cum[zlag + k] += np.dot(z[:nsamp - k], x[k:])   # m = +k

    y_cum *= scale / nrecord
    return y_cum

def cum4est(y, maxlag, nsamp, overlap, flag='biased', k1=0, k2=0):
    """
    Estimate fourth-order cumulant slice: C4(m, k1, k2) = cum(x*(t), x(t+m), x(t+k1), x*(t+k2))

    Parameters
    ----------
    y : np.ndarray
        Input signal (1D, real or complex).
    maxlag : int
        Maximum lag (output computed from -maxlag to +maxlag).
    nsamp : int
        Segment length.
    overlap : float
        Percentage overlap (0-100).
    flag : str
        'biased' or 'unbiased'.
    k1, k2 : int
        Fixed lags for the fourth-order slice.

    Returns
    -------
    y_cum : np.ndarray
        Estimated fourth-order cumulant slice C4(m, k1, k2), for -maxlag <= m <= maxlag.
    """
    y = y.flatten()
    N = len(y)
    overlap0 = overlap
    overlap = int(overlap / 100 * nsamp)
    nadvance = nsamp - overlap
    nrecord = (N - overlap) // nadvance

    nlags = 2 * maxlag + 1
    zlag = maxlag
    mlag = max(maxlag, abs(k1), abs(k2), abs(k1 - k2))
    mlag1 = mlag

    y_cum = np.zeros(nlags, dtype=np.complex128)

    # Unbiased or biased scaling
    if flag.lower().startswith('b'):
        scale = np.ones(nlags) / nsamp
    else:
        ind = np.arange(-maxlag, maxlag + 1)
        kmin = min(0, k1, k2)
        kmax = max(0, k1, k2)
        scale = nsamp - np.maximum(ind, kmax) + np.minimum(ind, kmin)
        scale = 1.0 / scale

    for i in range(nrecord):
        start = i * nadvance
        end = start + nsamp
        if end > N:
            break

        x = y[start:end]
        x = x - np.mean(x)
        cx = np.conj(x)
        z = np.zeros_like(x)

        # Compute x(n)*conj(x(n+k1))*x(n+k2)
        if k1 >= 0:
            z[:nsamp - k1] = x[:nsamp - k1] * cx[k1:]
        else:
            z[-k1:] = x[-k1:] * cx[:nsamp + k1]

        if k2 >= 0:
            z[:nsamp - k2] *= x[k2:]
        else:
            z[-k2:] *= x[:nsamp + k2]
        z[min(nsamp - abs(k2), nsamp):] = 0

        tmp = np.zeros(nlags, dtype=np.complex128)
        tmp[zlag] += np.dot(z, x)
        for k in range(1, maxlag + 1):
            tmp[zlag - k] += np.dot(z[k:], x[:nsamp - k])
            tmp[zlag + k] += np.dot(z[:nsamp - k], x[k:])

        tmp *= scale
        y_cum += tmp

        # Moment correction
        R_yy = cum2est(x, mlag, nsamp, overlap0, flag)
        m1 = mlag1 + k1
        m2 = mlag1 - k2
        m3 = mlag1 + (k1 - k2)

        R1 = R_yy[m1]
        R2 = R_yy[m3]
        R_seg = R_yy[mlag1 - maxlag:mlag1 + maxlag + 1]

        if np.iscomplexobj(x):
            M_yy = cum2x(np.conj(x), x, mlag, nsamp, overlap0, flag)
        else:
            M_yy = R_yy

        M_seg = M_yy[mlag1 - maxlag:mlag1 + maxlag + 1]
        M1 = M_yy[mlag1 + k2]

        y_cum -= R1 * R_seg
        y_cum -= R2 * R_seg
        y_cum -= M1.conj() * M_seg

    y_cum /= nrecord
    return y_cum