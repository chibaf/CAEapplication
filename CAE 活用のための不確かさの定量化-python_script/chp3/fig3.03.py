#===============================================================================
#   図　3.3  ヒストグラム
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   ='xx-large'

# %%
n = np.array([100, 1000, 10000])
b = np.array([ 10,   30,   100])

fig = plt.figure(figsize=(12.0, 4.0), dpi=100)
np.random.seed(1234)
for i in range(3):
    ns = np.random.randn(n[i])
    ax = plt.subplot(1, 3, i+1)
    ax.hist(ns, bins=b[i], color='grey', alpha=0.3)
    ax.hist(ns, bins=b[i], histtype='step', color='red')
    ax.set_title(f'$n=${n[i]}')
    ax.set_xlabel('$x$')
    ax.set_xticklabels('')
    ax.set_xticks([])
plt.show()