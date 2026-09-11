#===============================================================================
#   図　4.2 sinc関数による補間
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
plt.rcParams['font.size']        = 12
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
ax.plot(x, np.sinc(x), color='b')
ax.grid(linestyle='dotted')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$', rotation='horizontal')
ax.set_title(r'$y$ = sinc$(x) = \sin(\pi x)/(\pi x)$', fontsize='xx-large')
#
ax = plt.subplot(gs.new_subplotspec((0, 1), 1, 1))
y  = np.zeros(len(x))
for i in r:
    y = y + w[i] * np.sinc(x - h[i])
    ax.plot(x, w[i] * np.sinc(x - h[i]), color='b', linestyle='dotted', linewidth=0.5)
    ax.scatter(h[i], w[i], marker='x', color='r', s=50, linewidth=2, zorder=3)
ax.plot(x, y, color='b',)
ax.grid(linestyle='dotted')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\hat{y}$', rotation='horizontal')
ax.set_title(r'Interpolation by sinc$(x)$', fontsize='xx-large')
plt.show()