#===============================================================================
#   図　7.12  MCMC法による予測
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import GPy
import matplotlib.pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 'xx-large'
plt.rc('text.latex', preamble=r'\usepackage{upgreek, bm}')

# %%
xtrain  = np.array([ 0.00,  0.22,  0.44,  0.67,  0.90,  0.16,  0.34,  0.50,  0.73,  1.00 ])
ytrain  = np.array([-0.06,  0.97,  0.25, -0.9 , -0.53,  0.94,  0.85,  0.09, -0.93,  0.08 ])

# %%
ker    = GPy.kern.RBF(1, **{'name':'RBF'})
model  = GPy.models.GPRegression(xtrain[:, None], ytrain[:, None], **{'kernel':ker, 'noise_var':0.1})
# 
nsample  = 1000
stepsize = 0.01
np.random.seed(9876)
mcmc    = GPy.inference.mcmc.HMC(model, **{'stepsize':stepsize})
samples = mcmc.sample(nsample)
# 
p_name      = model.parameter_names()
log_lik = np.empty(nsample)
for i in range(nsample):
    for j in range(3):
        model[p_name[j]] = samples[i, j]
    log_lik[i] = model.log_likelihood()
argmax = np.argmax(log_lik)
# %%
cc  = ['blue', 'red', 'green', 'purple']
fig = plt.figure(figsize=(10.0, 5.0), dpi=100, tight_layout=True)
ax  = fig.add_subplot(2, 2, 1)
ax.plot(range(nsample), log_lik, color=cc[3], linewidth=0.5, label=r'$%.4f$' % log_lik[argmax])
ax.plot([argmax, argmax], ax.get_ylim(), color='grey', linewidth=0.5)
ax.plot([200, 1000], [log_lik[argmax], log_lik[argmax]], color='grey', linewidth=0.5)
plt.text(0,log_lik[argmax],r'$%.4f$' % log_lik[argmax], va='center')
plt.title(r'${\cal L}(\bm{\upvartheta})_{\rm max}$')
#
for i in range(3):
    ax = fig.add_subplot(2, 2, i + 2)
    ax.semilogy(range(nsample), samples[:, i], color=cc[i], linewidth=0.5)
    ax.plot(range(nsample), log_lik, color=cc[3], label=r'%.4f' % samples[argmax][i])
    ax.plot([argmax, argmax], ax.get_ylim(), color='grey', linewidth=0.5)
    ax.plot([180, 1000], [samples[argmax][i], samples[argmax][i]], color='grey', linewidth=0.5)
    plt.title(r'%s' % p_name[i])
    ylim = ax.get_ylim()
    ax.text(0,samples[argmax][i], r'%.4f' % samples[argmax][i], va='center')
plt.show()