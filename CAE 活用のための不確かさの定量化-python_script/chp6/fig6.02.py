#===============================================================================
#   図　6.2  θ2=1,0.1,0.01　に対応したカーネル行列と予測分布の変化
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
plt.rcParams['font.size']        = 12
plt.rc('text.latex', preamble=r'\usepackage{upgreek, bm}')

# %%
def Kernel(x1, x2, scale, sig_f):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    K  = np.empty((n1, n2))
    for row in range(n1):
        for col in range(n2):
            d = x1[row] - x2[col]
            K[row, col] = np.exp(- 0.5 * d * d / scale ** 2)
    return sig_f ** 2 * K

def log_likely(K, y_train):
    n   = y_train.shape[0]
    ll  = np.ravel(np.exp(- y_train.T * K.I * y_train[:, None] / 2))
    ll /= np.sqrt(np.linalg.det(K))
    ll /= np.sqrt((2.0 * np.pi) ** n)
    return np.log(ll)[0]

# %%
para_scale = np.sqrt(np.array([1.0, 0.1, 0.01]) / 2)
scale      = np.sqrt(0.5)
sig_n      = 0.1
sig_f      = 1.0

# %%
x_train = np.array([ 0.00, 0.22, 0.44,  0.67,  0.90, 0.16, 0.34, 0.50,  0.73, 1.00])
y_train = np.array([-0.06, 0.97, 0.25, -0.90, -0.53, 0.94, 0.85, 0.09, -0.93, 0.08])
x_test  = np.linspace(0.0, 1.0, 201)
X1, X2  = np.meshgrid(x_test, x_test)
n1      = x_train.shape[0]
n2      = x_test. shape[0]
    
# %%
n = 10
for para in range(len(para_scale)):
    scale  = para_scale[para]
    K     = Kernel(x_train, x_train, scale, sig_f) + sig_n ** 2 * np.mat(np.eye(n1))
    Kinv  = K.I
    K12   = Kernel(x_test, x_train, scale, sig_f)
    K21   = K12.T
    K22   = Kernel(x_test, x_test, scale, sig_f)
    mean  = np.ravel(K12 * Kinv * y_train[:, None])
    Var   = K22 - K12 * Kinv * K21
    sigma = np.sqrt(np.diag(Var))
       
    # %%    
    fig = plt.figure(figsize=(10.0, 4.0), dpi=100, tight_layout=True)
    gs  = GridSpec(1, 5)
    ax  = plt.subplot(gs.new_subplotspec((0, 0), 1, 2))
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.15)
    map = ax.contourf(X1, X2, Var)
    ax.scatter(x_train, x_train, marker='x', s=25, linewidth=1.5, color='w')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(r'$k_{\bm{\upvartheta}}({\bf x},{\bf x}^{\prime})$')
    fig.colorbar(map, cax=cax)
    #
    ax = plt.subplot(gs.new_subplotspec((0, 2), 1, 3))
    ax.plot(x_test, mean, label='mean')
    ax.fill_between(x_test, mean + 2.0 * sigma, mean - 2.0 * sigma, color='red', alpha=0.1, label=r'$\pm 2\sigma$')
    ax.scatter(x_train, y_train,
                   marker='x', color='r', s=50, linewidth=2, label='training data', zorder=3)
    ax.set_xlabel('$x$', fontsize='xx-large')
    ax.set_title(r'Gaussian Process Regression')
    ax.set_ylim([-1.5, 1.5])
    ax.grid()
    #    
    for nsample in range(10):
        sample = np.random.multivariate_normal(mean, Var)
        ax.plot(x_test, sample, linestyle='dotted')
    plt.legend(loc='upper right')
    #
    frm = r'$k_{\bm{\upvartheta}}({\bf x},{\bf x}^{\prime})= \theta_1\exp\left(-\|{\bf x}-{\bf x}^{\prime}\|^2/\theta_2\right)+\theta_3\delta({\bf x},{\bf x}^{\prime})$'
    fig.suptitle(frm+'\n'+r'$\theta_1 = %5.3f , \qquad\theta_2 = %5.3f , \qquad\theta_3 = %5.3f \qquad  {\cal L}(\bm{\upvartheta})=%6.3f$' % (sig_f**2, 2*scale**2, sig_n**2, log_likely(K, y_train)))
    #
    plt.show()