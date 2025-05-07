# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:51:21 2025

@author: Giannis
"""

import numpy as np
import matplotlib.pyplot as plt
from hosa_toolbox.cumulants import cum2est, cum3est, cum4est

np.random.seed(0)
x = np.random.randn(3000)
x_nl = x**2 - np.mean(x**2)

c2 = cum2est(x, maxlag=30, nsamp=300, overlap=50, flag='unbiased')
c3 = cum3est(x_nl, maxlag=20, nsamp=300, overlap=50, flag='unbiased', k1=2)
c4 = cum4est(x_nl, maxlag=20, nsamp=300, overlap=50, flag='unbiased', k1=0, k2=0)

lags = np.arange(-30, 31)
plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.plot(lags, c2)
plt.title("Second-Order Cumulant")
plt.grid()

plt.subplot(132)
lags = np.arange(-20, 21)
plt.plot(lags, np.real(c3), label="Real")
plt.plot(lags, np.imag(c3), label="Imag", linestyle='--')
plt.title("Third-Order Cumulant (k1=2)")
plt.legend()
plt.grid()

plt.subplot(133)
plt.plot(lags, c4)
plt.title("Fourth-Order Cumulant (k1=0, k2=0)")
plt.grid()
plt.tight_layout()
plt.show()
