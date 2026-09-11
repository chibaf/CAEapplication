#===============================================================================
#   図　7.9  予測分布から得られるランダムサンプリング
#
#   Copyright (c) 2024, ytrainuukou TOytrainONORI
#   All rights reserved.
#===============================================================================
import numpy as np
import GPy
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['figure.dpi']       = 200

# %% 
xtrain = np.array([ 0.00,  0.22,  0.44,  0.67,  0.90,  0.16,  0.34,  0.50,  0.73,  1.00 ])
ytrain = np.array([-0.06,  0.97,  0.25, -0.9 , -0.53,  0.94,  0.85,  0.09, -0.93,  0.08 ])

# %%
m = GPy.models.GPRegression(xtrain[:, None], ytrain[:, None])
m.optimize()
fig_dict = m.plot()
##
num_sample = 5
xnew       = np.linspace(-0.2, 1.2, 201)[:, None]
mean, cov  = m.predict_noiseless(xnew, full_cov=True)
mean       = mean.flatten()

np.random.seed(0)
sample = np.random.multivariate_normal(mean, cov, size=num_sample)
for i in range(num_sample):    
    plt.plot(xnew, sample[i, :], linewidth=2.0, linestyle='dotted')
plt.show()

# %% modify default collection
#
dp_collection = fig_dict['dataplot'][0]
mu_collection = fig_dict['gpmean'][0][0]
ci_collection = fig_dict['gpconfidence'][0]

dp_collection.update({'color':'red',  'linewidth':2, 'label':'training data', 'zorder':3})
mu_collection.update({'color':'blue', 'linewidth':1, 'label':'mean', 'zorder':2})
ci_collection.update({'color':'red', 'alpha':0.1, 'label':r'95\% CI'})
#
ax = dp_collection.axes
ax.legend(loc='lower right', fontsize='smaller')
ax.grid()
ax.set_title('Random sampling from predicted distribution', fontsize=16)
#
fig = dp_collection.get_figure()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
fig.savefig('fig7.9.png')
plt.imshow(plt.imread('fig7.9.png'))
plt.axis('off')
plt.show()