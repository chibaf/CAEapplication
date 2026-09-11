#===============================================================================
#   図　5.3 2次元カーネル関数（l=2の場合）
#   図　5.4 2次元カーネル関数（l=5の場合）
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 14
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
def mean_function(x):
    return np.zeros_like(x)

def covariance_function(xy, scale):
    n = xy.shape[0]
    return np.exp([[-np.dot(xy[i] - xy[j], xy[i] - xy[j]) / scale ** 2 for i in range(n)] for j in range(n)])

# %%
n   = 21
x   = np.linspace(-10.0, 10.0, n)
y   = np.linspace(-10.0, 10.0, n)
xyi = np.linspace(0, n ** 2 - 1, n ** 2)

xx, yy = np.meshgrid(x, y)
xy     = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

# %%
np.random.seed(0)
for scale in [2.0, 5.0]:
    covariance_matrix = covariance_function(xy, scale)
    fig = plt.figure(figsize=(12.0, 9.0), dpi=100, tight_layout=True)
    ax = plt.subplot(2, 2, 1)
    plt.contourf(xyi, xyi, covariance_matrix)
    ax.invert_yaxis()
    ax.set_xlabel(r'${\bf x}$')
    ax.set_ylabel(r'${\bf x}^{\prime}$', rotation = 'horizontal')
    ax.set_title(r'Kernel matrix     $\ell$=%d' % scale, fontsize='x-large')
    ax.set_aspect('equal')
    plt.xticks([0, n ** 2 / 2, n ** 2], [r'$(x_{1},y_{1})$','$(x_{i},y_{j})$', '$(x_{n},y_{n})$'])
    plt.yticks([0, n ** 2 / 2, n ** 2], [r'$(x_{1},y_{1})$','$(x_{i},y_{j})$', '$(x_{n},y_{n})$'])
    plt.colorbar()
    #
    for i in range(3):
        ax = fig.add_subplot(2, 2, i + 2, projection='3d')
        sample = np.random.multivariate_normal(mean_function(xy[:, 0]), covariance_matrix).reshape((n, n))
        ax.plot_surface(xx, yy, sample, linewidth=0.5, cmap='jet')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(r'Random samplings from GP (%d)' % (i + 1), fontsize='x-large')
        ax.grid(linestyle='dotted')
    #
    plt.show()