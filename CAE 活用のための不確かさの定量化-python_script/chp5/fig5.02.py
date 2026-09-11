#===============================================================================
#   図　5.2 １次元カーネル関数（n=201の場合）
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 16
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
def mean_function(x):
    return np.zeros_like(x)

def covariance_function(x1, x2, scale):
    return np.exp(-(x1 - x2) ** 2 / scale ** 2)

# %%
x      = np.linspace(-10.0, 10.0, 201)
x1, x2 = np.meshgrid(x, x)
np.random.seed(0)

# %%
m = mean_function(x)

fig = plt.figure(figsize=(16.0, 10.0), dpi=100, tight_layout=True)
#
n = 1
for scale in [0.1, 1.0, 5.0]:
    covariance_matrix = covariance_function(x1, x2, scale)
    ax = plt.subplot(3, 2, n)
    plt.contourf(x1, x2, covariance_matrix)
    ax.invert_yaxis()
    ax.set_xlabel(r'${\bf x}$')
    ax.set_ylabel(r'${\bf x}^{\prime}$', rotation ='horizontal')
    ax.set_title(r'$\ell$=%3.1f' % scale, fontsize='x-large')
    ax.set_aspect('equal')
    plt.colorbar()
    plt.xticks([-10.0, 0.0, 10.0], [r'$x_1$', '$x_{i}$', '$x_n$'])
    plt.yticks([-10.0, 0.0, 10.0], [r'$x_1$', '$x_{i}$', '$x_n$'])
    #
    ax = plt.subplot(3, 2, n+1)
    for k in range(n):
        sample = np.random.multivariate_normal(m, covariance_matrix)
        plt.scatter(x, sample, marker='x', label=f'Sample {k + 1}')
    plt.legend(loc='upper right')
    ax.set_xlabel(r'$x$')
    ax.set_title(r'Random samplings from Gaussian Process', fontsize='x-large')
    ax.grid(linestyle='dotted')
    #
    n += 2
plt.show()