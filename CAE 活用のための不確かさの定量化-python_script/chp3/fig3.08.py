#===============================================================================
#   図　3.8  正則化と過学習の抑制
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
def Erms (X, Y):
    return np.sqrt(np.mean((X - Y) ** 2))

# %%
xtrain = np.array([ 0.02, 0.22, 0.44,  0.67,  0.90, 0.16, 0.34, 0.50,  0.73, 1.00 ])
ytrain = np.array([-0.02, 0.97, 0.25, -0.90, -0.53, 0.94, 0.85, 0.12, -0.93, 0.08 ])

# %%
xgrid  = np.linspace(0.0, 1.0, 101)
y      = np.sin(2.0 * np.pi * xgrid)
#
nt  = len(xtrain)
ng  = len(xgrid)

# %%
deg = 9
#
Xd = np.mat(np.empty((nt, deg+1)))
Xg = np.mat(np.empty((ng, deg+1)))
for i in range(deg+1):
    Xd[:, i] = np.power(xtrain[:, None], i)
    Xg[:, i] = np.power(xgrid [:, None], i)    
# %%
fig, ax = plt.subplots(dpi=100)
ax.scatter(xtrain, ytrain, marker='x', color='k', linewidth=2, label='training Data', zorder=3)
ax.plot(xgrid, y, linestyle='dotted', label=r'$y\,=\,\sin\,(2\pi\,x)$')
#
lambd      = [0, 1e-4, 1e-2]
linestyle  = ['solid', 'dashdot', 'dashed']
for i in range(3):
    w = (Xd.T * Xd + lambd[i] * np.eye(deg+1)).I * Xd.T * ytrain[:, None]
    E = Erms(np.array(Xd * w), np.array(ytrain[:, None]))        
    ax.plot(xgrid, Xg * w, label=r'$\lambda=%.2E$   $E_{\rm rms}=%.3f$' % (lambd[i], E), linestyle=linestyle[i])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$', rotation='horizontal')
ax.set_ylim([-1.5, 1.5])
ax.grid(linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize='small')
ax.set_title('Overfitting and Regularization  ( M=$%d$ )' % deg)
plt.show()