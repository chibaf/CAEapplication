#===============================================================================
#   図　5.5  線形カーネル
#   図　5.6  多項式カーネル
#   図　5.7  指数カーネル
#   図　5.8  マターン３カーネル
#   図　5.9  マターン５カーネル
#   図　5.10 ガウスカーネル
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 14
plt.rcParams['axes.labelsize']   = 'xx-large'

# %% 
def mean_function(x):
    return np.zeros_like(x)

def covariance_function(kernel, x1, x2, para):
    nn = np.shape(x1)[0]
    print(kernel, para, nn)
    K = np.zeros((nn, nn))
    if kernel == 'Gaussian':
        for row in range(nn):
            for col in range(row, nn):
                K[row, col] = K[col, row] = np.exp(- np.dot(x1[row] - x2[col], x1[row] - x2[col]) / para)
    elif kernel == 'Linear':
        for row in range(nn):
            for col in range(row, nn):
                K[row, col] = K[col, row] = np.dot(x1[row], x2[col]) 
    elif kernel == 'Polynomial':
        for row in range(nn):
            for col in range(row, nn):
                K[row, col] = K[col, row] = (np.dot(x1[row], x2[col]) + 1) ** para
    elif kernel == 'Exponential':
        for row in range(nn):
            for col in range(row, nn):
                K[row, col] = K[col, row] = np.exp(- np.sqrt(np.dot(x1[row] - x2[col], x1[row] - x2[col])) / para)
    elif kernel == 'Matern3':
        c1 = np.sqrt(3.0)
        for row in range(nn):
            for col in range(row, nn):
                d = np.sqrt(np.dot(x1[row] - x2[col], x1[row] - x2[col])) / para
                K[row, col] = K[col, row] = (1 + c1 * d) * np.exp(- c1 * d)
    elif kernel == 'Matern5':
        c1 = np.sqrt(5.0)
        c2 = 5.0 / 3.0
        for row in range(nn):
            for col in range(row, nn):
                d = np.sqrt(np.dot(x1[row] - x2[col], x1[row] - x2[col])) / para
                K[row, col] = K[col, row] = (1 + c1 * d + c2 * d ** 2) * np.exp(- c1 * d)
    #
    return K
 
# %%
kernel_name = ['Linear', 'Polynomial', 'Exponential', 'Matern3', 'Matern5','Gaussian']
para_lin    = [1, 1, 1]
para_deg    = [1, 2, 3]
para_scale  = [0.05, 0.2, 0.5]
para        = [para_lin, para_deg, para_scale, para_scale, para_scale, para_scale]
       
# %%
n1     = 201
x      = np.linspace(-1.0, 1.0, n1)
x1, x2 = np.meshgrid(x, x)
#
n2     = 21
xx     = np.linspace(-1.0, 1.0, n2)
xx, yy = np.meshgrid(xx, xx)
xyi    = np.linspace(0, n2 ** 2 - 1, n2 ** 2)
xy     = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

# %%
np.random.seed(0)
for kernel in range(len(kernel_name)):
    fig = plt.figure(figsize=(16.0, 9.0), dpi=100, tight_layout=True)
    gs  = GridSpec(3, 5)
    fig.suptitle(r'Kernel matrix and random sampling from ' 
                 + kernel_name[kernel] + ' kernel function', fontsize='xx-large')
    for i in range(3):
        kernel_matrix= covariance_function(kernel_name[kernel], x, x, para[kernel][i])
        ax  = plt.subplot(gs.new_subplotspec((i, 0), 1, 1))
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.15)
        map = ax.contourf(x1, x2, kernel_matrix)
        ax.invert_yaxis()
        ax.set_xlabel(r'${\bf x}$')
        ax.set_ylabel(r'${\bf x}^{\prime}$', rotation = 'horizontal', labelpad=12)
        ax.set_aspect('equal')
        ax.grid(linestyle='dotted')
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['$x_0$', '$x_{i}$', '$x_n$'])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['$x_0$', '$x_{i}$', '$x_n$'])
        fig.colorbar(map, cax=cax)
        if kernel_name[kernel] == 'Linear':
            pass
        elif kernel_name[kernel] == 'Polynomial':
            ax.set_title(r'$M$ = %d' % para[kernel][i], fontsize='x-large')
        else:
            ax.set_title(r'$\theta$ = %4.2f' % para[kernel][i], fontsize='x-large')
        #
        ax = plt.subplot(gs.new_subplotspec((i, 1), 1, 2))
        for k in range(5):
            sample = np.random.multivariate_normal(mean_function(x), kernel_matrix)
            plt.plot(x, sample)
        ax.set_xlabel(r'$x$')
        ax.grid(linestyle='dotted')
        #
        kernel_matrix= covariance_function(kernel_name[kernel], xy, xy, para[kernel][i])
        ax  = plt.subplot(gs.new_subplotspec((i, 3), 1, 1))
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.15)
        map = ax.contourf(xyi, xyi, kernel_matrix)
        ax.invert_yaxis()
        ax.set_xlabel(r'${\bf x}$')
        ax.set_ylabel(r'${\bf x}^{\prime}$', rotation = 'horizontal', labelpad=12)
        ax.set_aspect('equal')
        ax.set_xticks([0, n2 ** 2 / 2, n2 ** 2])
        ax.set_xticklabels(['$(x_{0},y_{0})$','$(x_{i},y_{j})$', '$(x_{n},y_{n})$'])
        ax.set_yticks([0, n2 ** 2 / 2, n2 ** 2])
        ax.set_yticklabels(['$(x_{0},y_{0})$','$(x_{i},y_{j})$', '$(x_{n},y_{n})$'])
        fig.colorbar(map, cax=cax)
        #
        sample = np.random.multivariate_normal(mean_function(xy[:,0]), kernel_matrix).reshape((n2, n2))
        ax = plt.subplot(gs.new_subplotspec((i, 4), 1, 1), projection='3d')
        ax.plot_surface(xx, yy, sample, linewidth=0.5, cmap='jet')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.grid(linestyle='dotted')
    #
    plt.show()
