#===============================================================================
#   図　8.9  乱数発生の種を変化させたときの実験計画法の比較
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import GPy
import matplotlib.pyplot as plt
import pyDOE3 as doe

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
def branin(X):
    y  = (X[:, 1] - 5.1 / (4.0 * np.pi ** 2) * X[:, 0] ** 2 + 5.0 * X[:, 0] / np.pi - 6.0) ** 2
    y += 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(X[:, 0]) + 10.0
    return(y)

def LHS(n, seed):
    lhs = doe.lhs(2, samples=n, criterion='cm', iterations=10, random_state=seed)
    lhs = 15.0 * lhs
    lhs[:, 0] = lhs[:, 0] - 5.0
    return(lhs)

# %%
xg1          = np.linspace(-5.0, 10.0, 151)
xg2          = np.linspace( 0.0, 15.0, 151)
bf_x1, bf_x2 = np.meshgrid(xg1,xg2)
bf_X         = np.vstack((bf_x1.flatten(), bf_x2.flatten())).T
bf_y         = branin(bf_X).reshape((151, 151))

# %%
kg        = GPy.kern.RBF (input_dim=2, ARD=True)
kb        = GPy.kern.Bias(input_dim=2)
seed_list = [6789, 4321, 567, 7654]
ngrid     = [5, 7, 10, 20]

# %%
mins_list = np.array([])
vars_list = np.array([])
fig = plt.figure(figsize=(12.0, 6.0), dpi=100, tight_layout=True)       
gs  = plt.GridSpec(2, 4)
for iseed in range(4):
    seed = seed_list[iseed]
    th1_list  = np.array([])
    th2_list  = np.array([])
    th3_list  = np.array([])
    th4_list  = np.array([])
    for samp in [1,2]:
        for ntrn in ngrid:
            np.random.seed(seed)
            xg1   = np.linspace(-5.0, 10.0, ntrn)
            xg2   = np.linspace( 0.0, 15.0, ntrn)
            x1,x2 = np.meshgrid(xg1,xg2)
            if samp == 0:
                X = np.vstack((x1.flatten(),x2.flatten())).T
            if samp == 1:
                X = np.random.uniform(size=(ntrn**2, 2)) * 15.0
                X[:, 0]= X[:, 0] - 5.0
            if samp == 2:
                X = LHS(ntrn ** 2, seed)
            Y     = branin(X)[:,None]
            #
            k  = kg + kb
            m  = GPy.models.GPRegression(X, Y, k, normalizer=True)
            m.sum.rbf.variance.constrain_bounded     (1e-3, 1e6, warning=False)
            m.sum.bias.variance.constrain_bounded    (1e-3, 1e5, warning=False)
            m.sum.rbf.lengthscale.constrain_bounded  (1e-1, 1e3, warning=False)
            m.Gaussian_noise.variance.constrain_fixed(1e-6)
            m.optimize(max_iters=10000)
            th1_list = np.append(th1_list, m.sum.rbf.variance)
            th2_list = np.append(th2_list, m.sum.rbf.lengthscale[0])
            th3_list = np.append(th3_list, m.sum.rbf.lengthscale[1])
            th4_list = np.append(th4_list, m.sum.bias.variance)
            
    # %%
    lg = ['MCS','LHS']
    mk = ['o', 's']
    cl = ['red', 'green']
    #
    th1_list = np.sqrt(th1_list)
    th4_list = np.sqrt(th4_list)
    ax = plt.subplot(gs.new_subplotspec((0, iseed), 1, 1))  
    for ii in range(2):
        i = 4 * ii
        plt.plot(th1_list[i:i+4], th4_list[i:i+4], label=lg[ii], marker=mk[ii], color=cl[ii], alpha=.5)
    plt.loglog(base=10)
    ax.set_xlabel(r'$\sqrt{\theta_1}$', fontsize='x-large')
    ax.set_ylabel(r'$\sqrt{\theta_4}$', fontsize='x-large')
    ax.set_xlim([1, 200])
    ax.set_ylim([0.01, 200])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    plt.legend(loc='lower right')
    plt.title(r'random\_seed=$%d$' % seed_list[iseed])
    plt.grid()
    #
    ax = plt.subplot(gs.new_subplotspec((1, iseed), 1, 1))        
    for ii in range(2):
        i = 4 * ii
        plt.plot(th2_list[i:i+4], th3_list[i:i+4], marker=mk[ii], label=lg[ii], color=cl[ii], alpha=.5)
    ax.set_xlabel(r'$\sqrt{\theta_2/2}$', fontsize='x-large')
    ax.set_ylabel(r'$\sqrt{\theta_3/2}$', fontsize='x-large')
    ax.set_xlim([3.5, 6.0])
    ax.set_ylim([0, 150])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    plt.legend(loc='lower right')
    plt.grid()
    
plt.show()
