#===============================================================================
#   図　4.3 動径基底関数による補間
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
def RBF(x, mu, clen):
    return np.exp(- 0.5 * ((x - mu) / clen) ** 2)

# %%
x = np.linspace(-5.0, 5.0, 1001)
w = np.array([-0.5, -0.6, 0.3, 0.2, 0.5, 0.4, -0.3, 0.6, 0.2])
r = range(-4, 5, 1)
h = np.array(r)

# %%
fig = plt.figure(figsize=(10.0, 5.0), dpi=100, tight_layout=True)
gs  = GridSpec(1, 2) 
ax  = plt.subplot(gs.new_subplotspec((0, 0), 1, 1))
for i in r:
    ax.plot(x, w[i] * RBF(x, h[i], 1), color='b', linestyle='dotted', linewidth=1)
    ax.scatter(h[i], w[i], marker='x', color='r', zorder=3)
ax.grid(linestyle='dotted')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation='horizontal')
ax.set_title(r'Weighted RBF  ($\ell$=1)', fontsize='xx-large')
#
ax = plt.subplot(gs.new_subplotspec((0, 1), 1, 1))
for clen in [0.2, 0.4, 0.6, 0.8, 1.0]:
    y = np.zeros(len(x))
    for i in r:
        y = y + w[i] * RBF(x, h[i], clen)
        ax.scatter(h[i], w[i], marker='x', color='r', zorder=3)
    ax.plot(x, y, linewidth=1, label=r'$\ell$ = %3.1f' % clen)
ax.grid(linestyle='dotted')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\hat{y}$', rotation='horizontal')
ax.set_title(r'Interpolation by RBF', fontsize='xx-large')
plt.legend(loc='upper left')
plt.show()
