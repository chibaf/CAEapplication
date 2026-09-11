#===============================================================================
#   図　6.13  GPｙの最適化メソッドによるハイパーパラメータ系列と対数尤度の収束状況
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import GPy
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 16
plt.rcParams['axes.labelsize']   = 'xx-large'
plt.rc('text.latex', preamble=r'\usepackage{upgreek, bm}')

# %%
xtrain = np.array([ 0.00,  0.22,  0.44,  0.67,  0.90,  0.16,  0.34,  0.50,  0.73,  1.00 ])
ytrain = np.array([-0.06,  0.97,  0.25, -0.90, -0.53,  0.94,  0.85,  0.09, -0.93,  0.08 ])
ker    = GPy.kern.RBF(input_dim=1)
model  = GPy.models.GPRegression(xtrain[:, None], ytrain[:, None], kernel=ker)

# %%
iter_max = 100
sig_f = np.zeros(iter_max,)
sig_n = np.zeros(iter_max,)
s_len = np.zeros(iter_max,)
n_ll  = np.zeros(iter_max)
for iter in range(iter_max):
    sig_f[iter] = model.rbf.variance
    s_len[iter] = 2.0 * model.rbf.lengthscale ** 2
    sig_n[iter] = model.Gaussian_noise.variance
    n_ll[iter]  = - model.log_likelihood()
    model.optimize(optimizer='lbfgs', max_iters=1)

# %%
fig, ax = plt.subplots(figsize=(10.0, 7.0), dpi=100)
ax.semilogy(range(iter_max), sig_f, color='blue')
ax.semilogy(range(iter_max), s_len, color='red')
ax.semilogy(range(iter_max), sig_n, color='green')
ax.semilogy(range(iter_max), n_ll,  color='purple')
ax.set_xlabel('Iteration')
ax.grid()
ax.text(68, n_ll[iter_max-1]+0.4,    r'$-{\cal L}(\bm{\upvartheta})$ = %6.4f' % n_ll[iter_max-1], fontsize='x-large')
ax.text(75, sig_f[iter_max-1]+0.2,   r'$\theta_1$ = %6.4f' % sig_f[iter_max-1], fontsize='x-large')
ax.text(75, s_len[iter_max-1]+0.02,  r'$\theta_2$ = %6.4f' % s_len[iter_max-1], fontsize='x-large')
ax.text(75, sig_n[iter_max-1]+0.002, r'$\theta_3$ = %6.4f' % sig_n[iter_max-1], fontsize='x-large')
ax.set_title('Convergence of Hyperpameters (Gradient Method)', fontsize='x-large')
plt.show()