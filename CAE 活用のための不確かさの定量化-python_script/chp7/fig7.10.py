#===============================================================================
#   図　7.10  代表的なカーネル関数による予測
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import GPy
import matplotlib.pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = False

# %%
xtrain = np.array([ 0.00,  0.22,  0.44,  0.67,  0.90,  0.16,  0.34,  0.50,  0.73,  1.00 ])
ytrain = np.array([-0.06,  0.97,  0.25, -0.90, -0.53,  0.94,  0.85,  0.09, -0.93,  0.08 ])
# %%
kernel_dict = {'Exponential': GPy.kern.Exponential,
               'Matern32':    GPy.kern.Matern32,
               'Matern52':    GPy.kern.Matern52,
               'RBF':         GPy.kern.RBF,
               'Cosine':      GPy.kern.Cosine,
               'PeriodicExp': GPy.kern.PeriodicExponential}
kernel_keys = list(kernel_dict.keys())
# %%
fig = plt.figure(figsize=(12.0, 12.0), dpi=100, tight_layout=True)
for i in range(len(kernel_keys)):
    ax    = fig.add_subplot(3, 2, i + 1)
    ker   = kernel_dict[kernel_keys[i]](1)
    model = GPy.models.GPRegression(xtrain[:, None], ytrain[:, None], **{'kernel':ker})
    model.optimize()
    model.plot(**{'ax':ax, 'legend':False})
    model.plot_data(**{'ax':ax, 'color':'r', 'zorder':3})
    plt.ylim([-2.0, 2.0])
    plt.title(f'{kernel_keys[i]} \n AIC = ${2 * (-model.log_likelihood() + model.num_params) : .5f}$', fontsize=18)
    #
    np.random.seed(1234)
    num_sample = 10
    xnew       = np.linspace(-0.2, 1.2, 201).reshape((201, 1))
    sample     = model.posterior_samples_f(xnew, size=num_sample)
    for j in range(num_sample):
        plt.plot(xnew, sample[:, 0, j], linewidth=2.0, linestyle='dotted')
    for j in range(len(model.param_array)):
        plt.text(1.2, -1.4 - 0.18 * j, f'{model.parameter_names()[j]} = {model.param_array[j] : .6f}', ha='right')
plt.show()
