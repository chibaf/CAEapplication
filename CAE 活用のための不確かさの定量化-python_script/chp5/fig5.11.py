#===============================================================================
#   図　5.11  教師データ数に対応したカーネル行列と予測分布の変化
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 16
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
def Kernel(x1, x2, scale):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    K = np.empty((n1, n2))
    for row in range(n1):
        for col in range(n2):
            d = x1[row] - x2[col]
            K[row, col] = np.exp(- 0.5 * np.dot(d, d) / scale ** 2)
    return K

# %%
scale = np.sqrt(0.5)  
sig_n = 0.1 
sig_f = 1.0 

# %%
xtrain  = np.array([-1.0, -3.0,  0.0, 2.5, 4.0])
ytrain  = np.array([-5.0, -1.0, -2.0, 3.0, 1.0])
x_test  = np.linspace(-5.0, 5.0, 201)
X1, X2  = np.meshgrid(x_test, x_test)

# %%
fig = plt.figure(figsize=(10.0, 14.0), dpi=100, tight_layout=True)
gs = GridSpec(6, 5)
np.random.seed(0)
for n in range(len(xtrain) + 1):
    x_train = xtrain[0:n]
    y_train = ytrain[0:n]
    n1      = x_train.shape[0]
    n2      = x_test. shape[0]

    # %%
    K     = sig_f ** 2 * Kernel(x_train, x_train, scale) + sig_n ** 2 * np.mat(np.eye(n1))
    Kinv  = K.I
    K12   = Kernel(x_test, x_train, scale)
    K21   = K12.T
    K22   = Kernel(x_test, x_test, scale)
    Kinv  = K.I
    mean  = np.ravel(K12 * Kinv * y_train[:, None])
    Var   = K22 - K12 * Kinv * K21
    sigma = np.sqrt(np.diag(Var))

    # %%    
    ax = plt.subplot(gs.new_subplotspec((n, 0), 1, 2))
    ax.invert_yaxis()
    c = ax.contourf(X1, X2, Var, 15) 
    ax.scatter(x_train, x_train, marker='x', color='w', s=40, linewidth=2)
    ax.set_aspect('equal')
    plt.colorbar(c, ticks=np.arange(-0.25, 1.25, 0.25))
    ax.set_title(r'$n$ = %d' % n, fontsize='x-large')
    #
    ax = plt.subplot(gs.new_subplotspec((n, 2), 1, 3))
    ax.plot(x_test, mean, color='b', linewidth=1, label='mean')
    ax.fill_between(x_test, mean + 2.0 * sigma, mean - 2.0 * sigma, color='red', alpha=0.1, label=r'$\pm 2\sigma$')
    ax.scatter(x_train, y_train,
                   marker='x', color='r', s=50, linewidth=2, label='training data', zorder=3)
    ax.set_ylim([-6., 4.])
    ax.grid()
    #    
    for nsample in range(10):
        sample = np.random.multivariate_normal(mean, Var)
        ax.plot(x_test, sample, linestyle='dotted', linewidth=1.5)
    plt.legend(loc='lower right', fontsize='xx-small')
plt.show()