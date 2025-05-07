# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:51:54 2025

@author: Giannis
"""

import numpy as np
from hosa_toolbox.bicoherence import bicoher, plot_bicoher

N = 4096
t = np.arange(N)
x = np.cos(2*np.pi*0.2*t) + np.cos(2*np.pi*0.3*t)
x += 0.5 * x**2 + 0.3 * x**3 + 0.05 * np.random.randn(N)

bic, waxis = bicoher(x, nfft=64, wind='hann', nsamp=256, overlap=50)
plot_bicoher(bic, waxis)
