# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:52:19 2025

@author: Giannis
"""

import numpy as np
import matplotlib.pyplot as plt

N = 2048
t = np.arange(N)
x = np.cos(2*np.pi*0.15*t) + np.cos(2*np.pi*0.25*t)
x_nl = x + 0.6 * x**2 + 0.3 * x**3 + 0.05 * np.random.randn(N)

plt.plot(t[:300], x_nl[:300])
plt.title("Synthetic Nonlinear Signal (First 300 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
