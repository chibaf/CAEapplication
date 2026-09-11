#===============================================================================
#   図　7.11  収束状況（上：10回、中：20回、下：50回）
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
xtrain  = np.array([ 0.00,  0.22,  0.44,  0.67,  0.90,  0.16,  0.34,  0.50,  0.73,  1.00 ])
ytrain  = np.array([-0.06,  0.97,  0.25, -0.9 , -0.53,  0.94,  0.85,  0.09, -0.93,  0.08 ])
# %%
optimizer_dict = {'SCG':    GPy.inference.optimization.opt_SCG,
                  'BFGS':   GPy.inference.optimization.opt_bfgs,
                  'L-BFGS': GPy.inference.optimization.opt_lbfgsb,
                  'TNC':    GPy.inference.optimization.opt_tnc}
optimizer_keys = list(optimizer_dict.keys())
# %%
ip  = 1
fig = plt.figure(figsize=(16.0, 12.0), dpi=100, tight_layout=True)
for iter in [10, 20, 50]:
    for i in range(len(optimizer_keys)):
        ax    = fig.add_subplot(3, 4, ip)
        ker   = GPy.kern.RBF(1)
        model = GPy.models.GPRegression(xtrain[:, None], ytrain[:, None], **{'kernel':ker})
        model.optimize(**{'optimizer':optimizer_dict[optimizer_keys[i]](max_iters=iter)})
        model.plot(**{'ax':ax, 'legend':False})
        model.plot_data(**{'ax':ax, 'color':'r', 'zorder':3})
        plt.ylim([-2.0, 2.0])
        plt.grid()
        plt.title(f'{optimizer_keys[i]} \n AIC = {2.0 * (-model.log_likelihood()) + model.num_params : .5f}', fontsize=18)
        #
        np.random.seed(1234)
        xnew = np.linspace(-0.2, 1.2, 201).reshape((201,1))
        mean, cov = model.predict_noiseless(xnew, full_cov=True)
        mean = mean.flatten()
        sample = np.random.multivariate_normal(mean, cov, size=10)
        for j in range(10):
            plt.plot(xnew, sample[j,:], linewidth=2.0, linestyle='dotted')
        ip += 1
plt.show()