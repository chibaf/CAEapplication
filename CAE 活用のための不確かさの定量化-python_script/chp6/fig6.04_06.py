#===============================================================================
#   図　6.4  １回の繰り返しで得られた推定結果と予測分布
#   図　6.5  ３回の繰り返しで得られた推定結果と予測分布
#   図　6.6  ５回の繰り返しで得られた推定結果と予測分布
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import numpy as np
from numpy.linalg import inv, det
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 'xx-large'
plt.rc('text.latex', preamble=r'\usepackage{upgreek, bm}')

# %%
def kernel_RBF(params):
    return lambda x, y: params[0] * np.exp(- (x - y) ** 2 / params[1])

def kv(x, xt, kernel):
    return np.array([kernel(x, xi) for xi in xt])

def km(x, kernel):
    nk = len(x)
    return np.array([kernel(xi, xj) for xi in x for xj in x]).reshape(nk, nk)

# %% 
def PLOT2D(ax, ix, iy, iz, ind, inv):
    sc = ax.scatter(LG[ind[iz], ix], LG[ind[iz], iy], c=LL[ind[iz]], cmap='jet', alpha=0.5, vmin=LL.min(), vmax=LL.max())
    ax.scatter   (LGmax[ix], LGmax[iy], marker='x', color='k', s=50, linewidth=3, zorder=3)
    ax.set_xlim  ([min(logtheta[ix,:]), max(logtheta[ix,:])])
    ax.set_ylim  ([min(logtheta[iy,:]), max(logtheta[iy,:])])
    ax.set_xlabel(Label[ix], labelpad=0)
    ax.set_ylabel(Label[iy], labelpad=0)
    if inv:
        ax.invert_yaxis()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[step]))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[step]))
    ax.grid(linestyle='dotted', linewidth=1.5)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.2)
    cb  = plt.colorbar(sc, cax=cax)
    cb.formatter.set_useOffset(False)
    cb.update_ticks()

# %%
xtrain = np.array([ 0.00,  0.22,  0.44,  0.67,  0.90,  0.16,  0.34,  0.50,  0.73,  1.00 ])
ytrain = np.array([-0.06,  0.97,  0.25, -0.90, -0.53,  0.94,  0.85,  0.09, -0.93,  0.08 ])
xtest  = np.linspace(-0.2, 1.2, 71)
n1     = len(xtrain)
n2     = len(xtest)
eye1   = np.eye(n1)
eye2   = np.eye(n2)

# %% 
nd    = 41
LGmax = np.array([0.0, 0.0, 0.0])
span  = np.array([5.0, 5.0, 5.0])
ticks = np.array([2.0, 0.5, 0.1, 0.05, 0.01])
Level = -20.0
Label = [r'$\log \theta_1$', r'$\log \theta_2$', r'$\log \theta_3$']
#
for step in range(5):
    print('iteration ' + str( step+1) + '  ', end='')
    ls            = np.linspace(LGmax - span, LGmax + span, nd)
    mg1, mg2, mg3 = np.meshgrid(ls[:,0], ls[:,1], ls[:,2])
    logtheta      = np.vstack((mg1.flatten(), mg2.flatten(), mg3.flatten()))
    ng            = np.size(logtheta, axis=1)
    th            = 0.1
    # 
    LL  = LG = np.array([], dtype=np.float32)
    const  = np.sqrt(2.0 * np.pi) ** n1
    for i in range(ng):
        theta  = np.exp(logtheta[:, i])
        kernel = kernel_RBF([theta[0], theta[1]])
        K      = km(xtrain, kernel) + theta[2] * eye1
        ll     = np.exp(-(ytrain @ inv(K) @ ytrain.T) / 2.0) / np.sqrt(det(K))
        ll     = np.log(ll / const + 1e-32)
        #
        if ll > Level:
            LL = np.append(LL, ll)
            LG = np.append(LG, logtheta[:, i], axis=0)
        if i / ng > th:
            print('-', end='')
            th += 0.1
    # 
    LG    = LG.reshape((len(LL), 3))
    LGmax = LG[LL.argmax(), :]
    theta = np.exp(LGmax)
    print('> Theta =', theta, ',  LLmax =', LL.max())
    #
    ind = []
    for i in range(3):
        ind.append(np.nonzero(LG[:, i] == LGmax[i]))
        
    # %% 
    Level = (4.0 * LL.max() + LL.mean()) / 5.0
    span  = span / 4.0
    
# %% 
    kernel = kernel_RBF([theta[0], theta[1]])
    k11    = km(xtrain, kernel) + eye1 * theta[2]
    k22    = km(xtest,  kernel)
    k12    = kv(xtest,  xtrain, kernel)
    mean   = k12.T @ inv(k11) @ ytrain.T 
    var_f  = k22 - k12.T @ inv(k11) @ k12
    var_n  = var_f  + eye2 * theta[2] 
    sigma  = np.sqrt(np.diag(var_n))
    
    # %% 
    plt.figure(figsize=(10.0, 15.0), dpi=100, tight_layout=True)
    gs  = GridSpec(7, 2)
    #
    size = 0.5
    ix = 1;  iy = 2;  iz = 0
    ax = plt.subplot(gs.new_subplotspec((1, 0), 2, 1), projection='3d')
    ax.scatter   (LG[:, ix], LG[:, iy], LG[:, iz], c=LL, cmap='jet', alpha=0.15 - 0.02 * step, s=size, vmin=LL.min(), vmax=LL.max())
    ax.scatter   (LGmax[ix], LGmax[iy], LGmax[iz], marker='x', color='k', s=50, linewidth=3)
    ax.set_xlabel(Label[ix], labelpad=3)
    ax.set_ylabel(Label[iy], labelpad=5)
    ax.set_zlabel(Label[iz], labelpad=5)
    ax.set_xlim  ([min(logtheta[ix,:]), max(logtheta[ix,:])])
    ax.set_ylim  ([min(logtheta[iy,:]), max(logtheta[iy,:])])
    ax.set_zlim  ([min(logtheta[iz,:]), max(logtheta[iz,:])])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ticks[step]))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ticks[step]))
    
    # %%
    PLOT2D(plt.subplot(gs.new_subplotspec((1, 1), 2, 1)), iy, ix, iz, ind, True)
    PLOT2D(plt.subplot(gs.new_subplotspec((3, 0), 2, 1)), ix, iz, iy, ind, False)
    PLOT2D(plt.subplot(gs.new_subplotspec((3, 1), 2, 1)), iy, iz, ix, ind, False)
    
    # %% 
    ax = plt.subplot(gs.new_subplotspec((5, 0), 2, 2))
    ax.plot        (xtest, mean, label='mean')
    ax.fill_between(xtest, mean + 2.0 * sigma, mean - 2.0 * sigma, facecolor='red', alpha=0.1, label=r'$\pm2\sigma$')
    ax.scatter     (xtrain, ytrain, marker='x', color='r', s=50, linewidth=2, label='training data', zorder=3)
    np.random.seed(1234)
    for nsample in range(10):
        sample = np.random.multivariate_normal(mean, var_f)
        ax.plot(xtest, sample, linestyle='dotted')
        
    # %%    
    ax.set_xlim  ([-0.2, 1.2])
    ax.set_ylim  ([-1.5, 1.5])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation='horizontal')
    ax.grid      (linestyle='dotted', color='grey')
    plt.legend(loc='lower right')
    
    # %%
    ax = plt.subplot(gs.new_subplotspec((0, 0), 1, 2))
    s1  = r'${\cal L}(\bm{\upvartheta})_{\rm max}=%.5f$' % LL.max()
    s2  = r'$\theta_1=%.6f$' % theta[0]
    s3  = r'$\theta_2=%.6f$' % theta[1]
    s4  = r'$\theta_3=%.6f$' % theta[2]
    ax.text(0.65, -0.2, s1 + '\n' + s2 + '\n'+ s3 + '\n' + s4, fontsize='xx-large', ha='right')
    ax.axis('off')
    
    # %%
    plt.show()