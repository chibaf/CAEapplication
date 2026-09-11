#===============================================================================
#   図　7.14  例：回帰モデルのプロット
#   図　7.15  AICが高い回帰モデルの例
#   図　7.16  標準化による最適回帰モデル
#   図　7.17  正規化による最適回帰モデル
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import pandas as pd
import GPy
import matplotlib.pyplot as plt
import datetime as dt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import os

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = False
plt.rcParams['font.size']        = 20
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%  ---------- Class definition:  Trining data ----------
class Training_data:
    def __init__(self):
        self.d_start = dt.date(1960, 1, 1)
        self.d_end   = dt.date.today()
        
    def set_train_data(self, x, y):
        self.dlim    = (self.d_start, self.d_end)
        self.xlim    = np.array((0, (self.d_end - self.d_start).days))
        self.ylim    = np.array((9.3, 10.3))
        n            = 500
        self.x       = pre_process(x, x)
        self.y       = pre_process(y, y)
        xn           = pre_process(self.xlim, x)
        self.xn      = [np.linspace(xn[0][0], xn[0][1], n).reshape(n, 1),
                        np.linspace(xn[1][0], xn[1][1], n).reshape(n, 1)]
        self.pp_list = ['Standardization', 'Normalization']

def pre_process(data, dtrain):
    return [(data - np.mean(dtrain)) / np.std(dtrain),
            (data - np.min(dtrain))  / (np.max(dtrain) - np.min(dtrain))]

# %%  ---------- Function definition:  Draw & Plot ----------
def get_cax(subplot):
    ax     = fig.add_subplot(subplot[0], subplot[1], subplot[2])
    ax_sub = make_axes_locatable(ax)
    cax    = ax_sub.append_axes("right", size="5%", pad=0.15)
    return (ax, cax)    

def plot_data(fig, subplot, x, y, xlim, title):
    ax = fig.add_subplot(subplot[0], subplot[1], subplot[2])
    ax.scatter(x, y, marker='x', color='b', label='training data', zorder=3)
    ax.set_title(title, fontsize='smaller')
    ax.set_xlim(xlim)
    ax.set_ylim(D.ylim)
    return ax
    
def draw_predict(fig, subplot, model, xd, xn, scale, offset, label):
    mean, cov = model.predict(xn, full_cov=True)
    mean      = mean.flatten()
    sigma     = np.sqrt(np.diag(cov))
    AIC       = - 2.0 * model.log_likelihood() + 2.0 * model.num_params
    samples   = model.posterior_samples_f (xn, size=5).reshape((len(xn), 5))
    ax        = plot_data(fig, subplot, xd, ytrain, [np.min(xn), np.max(xn)], 
                          label + f'  AIC = {AIC : .5f}')
    ax.plot(xn, mean    * scale + offset, color='g', label='mean')
    ax.plot(xn, samples * scale + offset, linestyle='dotted')
    ax.fill_between(xn.flatten(), ((mean + 2.0 * sigma) * scale + offset),
                                  ((mean - 2.0 * sigma) * scale + offset),
                                  color='r', alpha=0.1, label=r'$\pm 2\sigma$')
    ax.set_xlim([np.min(xn), np.max(xn)])
    ax.legend(loc='upper right', fontsize='smaller')
    p_name = model.parameter_names()
    p_val  = model.param_array
    pos_y  = D.ylim[0] + 0.07 * len(p_val) - 0.05
    for ms in range(len(p_val)):
        s_name = p_name[ms]
        ax.text(np.mean(xn), pos_y, 
             f'{s_name:s} = ${p_val[ms]:.4E}$', ha='right', fontsize='smaller')
        pos_y -= 0.07
    return (cov, AIC)

def draw_kernel(fig, subplot, K, label):
    ax, cax = get_cax(subplot)
    map     = ax.imshow(np.flipud(K))
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([0, 9, 19])
    ax.set_xticklabels(['$x_1$', '$x_{10}$', '$x_{20}$'])
    ax.set_yticks([])
    ax.set_title('Kernel covariance (' + label +')', fontsize='smaller')
    fig.colorbar(map, cax=cax)
    
def draw_covariance(fig, subplot, cov, xnew, xd):
    vmax    = np.max(np.abs(cov))
    vmin    = -vmax
    x1, x2  = np.meshgrid(xnew, xnew)
    ax, cax = get_cax(subplot)
    map     = ax.contourf(x1, x2, cov, cmap='seismic',
                         norm=Normalize(vmin=vmin, vmax=vmax))
    ax.scatter(xd, xd, color='b', marker='x', alpha=0.5, zorder=3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_title('Covariance of prediction', fontsize='smaller') 
    fig.colorbar(map, cax=cax)

# %%  ---------- Kernel definition ----------
ker_dict = {'Linear':      GPy.kern.Linear(input_dim=1),           # 0
            'Poly(3)':     GPy.kern.Poly(input_dim=1, order=3),    # 1
            'Exponential': GPy.kern.Exponential(input_dim=1),      # 2
            'Matern3':     GPy.kern.Matern32(input_dim=1),         # 3
            'Matern5':     GPy.kern.Matern52(input_dim=1),         # 4
            'RBF':         GPy.kern.RBF(input_dim=1)}              # 5
ker_keys = list(ker_dict.keys())
ker_list = [[2],     [3],     [4],     [5],
            [0,2],   [0,3],   [0,4],   [0,5],
            [1,2],   [1,3],   [1,4],   [1,5],
            [0,1,2], [0,1,3], [0,1,4], [0,1,5]]

# %%  ---------- Read training data & Create Class ----------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
data   = np.array(pd.read_excel('training_data.xlsx'))
num_d  = len(data[:, 0])
dtrain = data[:, 0].reshape(num_d, 1)
xtrain = data[:, 1].reshape(num_d, 1)
ytrain = data[:, 2].reshape(num_d, 1)
D      = Training_data()
D.set_train_data(xtrain, ytrain)
# %%  ---------- Begin main loop ----------
AIC       = np.empty((len(ker_list) + 1, 2))
name_list = []
idx       = 0

# %%  ---------- Create kernel function & name ----------
for kl in ker_list:
    Kernel   = ker_dict[ker_keys[kl[0]]]
    ker_name = ker_keys[kl[0]]
    for kn in range (len(kl) - 1):
        Kernel   += ker_dict[ker_keys[kl[kn + 1]]]
        ker_name += ' + ' +  ker_keys[kl[kn + 1]]  
    # %%  ---------- Plot training data ----------
    fig = plt.figure(figsize=(28.0, 16.0), dpi=100, tight_layout=True)
    fig.suptitle('World record: Men\'s 100m race \n'
                     + f'from {str(D.d_start):s} to {str(D.d_end):s}\n'
                     + 'Kernel : ' + ker_name, fontsize='x-large')
    plot_data(fig, (3, 5, (1, 2)), data[:,0], data[:,2], D.dlim,
              'Training data (xaxis: year-month-day)')
    plot_data(fig, (3, 5, (4, 5)), data[:,1], data[:,2], D.xlim,
              f'Training data (xaxis : duration from {str(D.d_start):s})')
    draw_kernel(fig, (3, 5, 3), Kernel.K(xtrain), 'w/o pre-process')  
    # %%  ---------- Draw Prediction / kernel ----------
    for pp in range(2):
        pp5   = 5 * pp
        Model = GPy.models.GPRegression(D.x[pp], D.y[0], kernel=Kernel)
        Model.param_array[:] = 1
        draw_kernel(fig, (3, 5, pp5 + 8), Kernel.K(D.x[pp]), 'before optimize')
        np.random.seed(1234)
        Model.optimize()
        draw_kernel(fig, (3, 5, pp5 + 9), Kernel.K(D.x[pp]), 'after optimize')  
        cov, AIC[idx, pp] = draw_predict(fig, (3 ,5 ,(pp5 + 6, pp5 + 7)), Model,
                                D.x[pp], D.xn[pp], 
                                np.std(ytrain), np.mean(ytrain), D.pp_list[pp])
        draw_covariance(fig, (3, 5, pp5 + 10), cov, D.xn[pp], D.x[pp])
    # %%  ---------- End of main loop ----------
    plt.show()
    name_list.append(ker_name)
    idx += 1

# %%  ---------- Save AIC profile to Excel----------
name_list.append('*** Optimal kernel ***')
AIC[idx, 0] = np.min(AIC[0:-1, 0])
AIC[idx, 1] = np.min(AIC[0:-1, 1])
df_AIC = pd.DataFrame({'kernel':name_list, 
                       'AIC(std)':AIC[:, 0], 'AIC(nrm)':AIC[:, 1]})
print('\n\nAIC =')
print(df_AIC)
df_AIC.to_excel('100m_world_record.xlsx')

