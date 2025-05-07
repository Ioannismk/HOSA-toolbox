# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:53:35 2025

@author: Giannis
"""

# tests/test_all.py
import numpy as np
from hosa_toolbox.cumulants import cum2est, cum3est, cum4est
from hosa_toolbox.bispectrum import bispecd
from hosa_toolbox.bicoherence import bicoher

def generate_signal(N=4096):
    t = np.arange(N)
    x = np.cos(2*np.pi*0.2*t) + np.cos(2*np.pi*0.3*t)
    x += 0.5 * np.cos(2*np.pi*0.5*t)  # nonlinearity target
    x += 0.6 * x**2 + 0.3 * x**3 + 0.05 * np.random.randn(N)
    return x

def test_cumulants():
    x = np.random.randn(2048)
    c2 = cum2est(x, maxlag=20, nsamp=256, overlap=50, flag='unbiased')
    c3 = cum3est(x, maxlag=10, nsamp=256, overlap=50, flag='biased', k1=2)
    c4 = cum4est(x, maxlag=10, nsamp=256, overlap=50, flag='biased', k1=1, k2=2)
    assert np.isfinite(c2).all(), "cum2est failed"
    assert np.isfinite(c3).all(), "cum3est failed"
    assert np.isfinite(c4).all(), "cum4est failed"
    print("✅ cumulant tests passed")

def test_bispectrum():
    x = generate_signal()
    B, _ = bispecd(x, nfft=64, wind=5, nsamp=256, overlap=50)
    assert np.isfinite(B).all(), "bispecd failed"
    print("✅ bispectrum test passed")

def test_bicoherence():
    x = generate_signal()
    bic, _ = bicoher(x, nfft=64, wind='hann', nsamp=256, overlap=50)
    assert np.isfinite(bic).all(), "bicoher failed"
    assert 0 <= bic.max() <= 1, "bicoher out of bounds"
    print("✅ bicoherence test passed")

if __name__ == "__main__":
    print("Running HOSA Toolbox Tests...")
    test_cumulants()
    test_bispectrum()
    test_bicoherence()
    print("🎉 All tests passed.")
