# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:51:40 2025

@author: Giannis
"""

import numpy as np
from hosa_toolbox.bispectrum import bispecd, plot_bispecd

N = 4096
t = np.arange(N)
x = np.cos(2*np.pi*0.2*t) + np.cos(2*np.pi*0.3*t) + 0.5*np.cos(2*np.pi*0.5*t)
B, w = bispecd(x, nfft=64, wind=5, nsamp=256, overlap=50)
plot_bispecd(B, w)
