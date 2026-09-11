#===============================================================================
#   図　6.7  対数尤度とハイパーパラメータ系列の推移
#   図　6.8  ハイパーパラメータ系列の3次元分布
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 16
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
def Kernel(x, x_prime, noise, thetas):
    return thetas[0] * np.exp(-(x - x_prime) ** 2 / thetas[1]) + noise * thetas[2]

def log_likely(params):
    train_length = len(x_train)
    K = np.zeros((train_length, train_length))
    for x_idx in range(train_length):
        for x_prime_idx in range(train_length):
            K[x_idx, x_prime_idx] = Kernel(x_train[x_idx], x_train[x_prime_idx], x_idx == x_prime_idx, params)
    y  = y_train
    yy = np.dot(np.linalg.inv(K), y_train)
    return - (10.0 * np.log(2.0 * np.pi) + np.linalg.slogdet(K)[1] + np.dot(y, yy)) / 2.0

# %%
def MCMC(x_train, y_train, bounds, initial_params=np.ones(3), n_iter=40000):
    params     = initial_params
    log_params = np.log(params)
    log_bounds = np.log(bounds)
    log_scale  = log_bounds[:, 1] - log_bounds[:, 0]
    #
    lml_prev   = log_likely(params)
    theta_list = np.array([])
    lml_list   = np.array([])
    # %%    
    for _ in range(n_iter):
        move = 1e-2 * np.random.normal(0, log_scale, size=len(params))
        need_resample = (log_params + move < log_bounds[:, 0]) | \
                        (log_params + move > log_bounds[:, 1])
        while(np.any(need_resample)):
            move[need_resample] = np.random.normal(0, log_scale, size=len(params))[need_resample]
            need_resample = (log_params + move < log_bounds[:, 0]) | \
                            (log_params + move > log_bounds[:, 1])
        # create proposed distribution
        next_log_params = log_params + move
        next_params     = np.exp(next_log_params)
        lml_next        = log_likely(next_params)
        r               = np.exp(lml_next - lml_prev)
        # metropolis update
        if r > 1 or r > np.random.random():
            params      = next_params
            log_params  = next_log_params
            lml_prev    = lml_next
            theta_list  = np.append(theta_list, params)
            lml_list    = np.append(lml_list, lml_prev)
    argmax     = np.argmax(lml_list)
    theta_list = theta_list.reshape(len(lml_list), 3)
    
    # %%
    color = ['blue', 'red', 'green']
    y_lim = [[1e-2, 1e2], [1e-2, 1.0], [1e-4, 1.0]]
    
    fig = plt.figure(figsize=(10.0, 10.0), dpi=100, tight_layout=True)
    gs  = plt.GridSpec(4, 4)
    ax  = plt.subplot(gs.new_subplotspec((0, 0), 1, 3))  
    ax.plot(range(len(lml_list)), lml_list,
            label=r'${\cal L}(\mathbf{\vartheta})=%8.4f$' % lml_list[argmax], color='purple', linewidth=0.5, alpha=0.5)
    ax.set_ylim([-12,-2])
    ax.plot([argmax, argmax], [-12.0, -2.0], color='k', lw=2)
    ax.legend(loc='lower left')
    #
    ax = plt.subplot(gs.new_subplotspec((0, 3), 1, 1))  
    plt.hist(lml_list, bins=100, color='purple', alpha=0.5, orientation='horizontal')#, np.log=True)
    ax.set_ylim([-12.0,-2.0])
    ax.set_yticklabels([])
    ax.set_yticks([])   
    #
    for i in range(3):
        ax = plt.subplot(gs.new_subplotspec((i + 1, 0), 1, 3))  
        ax.semilogy(range(len(lml_list)), theta_list[:,i], 
                label=r'$\theta_%d$ = %6.4f' % (i+1, theta_list[argmax,i]), color=color[i], linewidth=0.5, alpha=0.5)
        ax.set_ylim(y_lim[i])
        ax.plot([argmax, argmax], y_lim[i], color='k', lw=2)
        ax.legend(loc='lower left')
        #
        ax = plt.subplot(gs.new_subplotspec((i + 1, 3), 1, 1))  
        plt.hist(np.log10(theta_list[:,i]), bins=100, color=color[i], alpha=0.3, orientation='horizontal')
        ax.set_xlim([0, 1500])
        ax.set_ylim(np.log10(y_lim[i]))
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.plot([0, 1500], [np.log10(theta_list[argmax, i]), np.log10(theta_list[argmax, i])], color='k', lw=1.5)
#
    fig.suptitle('Gaussian Process Regression by MCMC', fontsize='x-large')
    plt.show()
    
    # %%
    fig = plt.figure(figsize=(11.0, 10.0), dpi=100)
    ax  = fig.add_subplot(1, 1, 1, projection='3d')
    mp  = ax.scatter(np.log(theta_list[:, 1]),
                     np.log(np.sqrt(theta_list[:, 2] / 2.0)),
                     np.log(theta_list[:, 0]), 
                     c=lml_list, cmap='jet', alpha=0.1, s=5,
                     vmin=lml_list.max() - 3.0, vmax=lml_list.max())
    ax.set_xlim([-4.0, 0.0])
    ax.set_ylim([-4.0, 0.0])    
    ax.set_zlim([-4.0, 2.0])    
    ax.set_xlabel(r'$\log \theta_2$', labelpad=15)
    ax.set_ylabel(r'$\log \theta_3$', labelpad=15)
    ax.set_zlabel(r'$\log \theta_1$', labelpad=10)
    plt.colorbar(mp, ax=ax, shrink=0.75, pad=0.1)
    plt.show()

# %%
x_train = np.array([ 0.00,  0.22,  0.44,  0.67,  0.90,  0.16,  0.34,  0.50,  0.73,  1.00 ])
y_train = np.array([-0.06,  0.97,  0.25, -0.90, -0.53,  0.94,  0.85,  0.09, -0.93,  0.08 ])
x_test  = np.linspace(-0.2, 1.2, 241)
#
np.random.seed(0)
MCMC(x_train, y_train, 
     bounds=np.array([[1e-4, 1e2], [1e-4, 1e2], [1e-4, 1e2]]), 
     initial_params=np.array([0.5, 0.5, 0.5]))