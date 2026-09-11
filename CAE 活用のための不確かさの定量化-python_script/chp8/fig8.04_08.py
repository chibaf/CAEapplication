#===============================================================================
#   図　8.4  Brain関数の3次元表面プロットと等高線プロット
#   図　8.5  正方格サンプリング
#   図　8.6  モンテカルロサンプリング
#   図　8.7  ラテン超方格サンプリング
#   図　8.8  ハイパーパラメータとグローバル最小値の収束状況
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import GPy
import matplotlib.pyplot as plt
import pyDOE3 as doe
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 18
plt.rcParams['axes.labelsize']   = 'xx-large'
plt.rc('text.latex', preamble=r'\usepackage{upgreek, bm}')

# %%
def branin(X):
    y  = (X[:, 1] - 5.1 / (4.0 * np.pi ** 2) * X[:, 0] ** 2 + 5.0 * X[:, 0] / np.pi - 6.0) ** 2
    y += 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(X[:, 0]) + 10.0
    return(y)

def LHS(n):
    lhs = doe.lhs(2, samples=n, criterion='cm', iterations=10, random_state=seed)
    lhs = 15.0 * lhs
    lhs[:, 0] = lhs[:, 0] - 5.0
    return(lhs)

# %%
seed = 6789
#
xg1         = np.linspace(-5.0, 10.0, 151)
xg2         = np.linspace( 0.0, 15.0, 151)
bf_x1,bf_x2 = np.meshgrid(xg1,xg2)
bf_X        = np.vstack((bf_x1.flatten(), bf_x2.flatten())).T
bf_y        = branin(bf_X).reshape((151, 151))
#
fig = plt.figure(figsize=(12.0, 6.0), dpi=100, tight_layout=False)
ax  = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel(r'$x_1$', size='xx-large')
ax.set_ylabel(r'$x_2$', size='xx-large')
ax.set_title('3D profile of Branin function', size='xx-large')
#
ax.plot_surface(bf_x1, bf_x2, bf_y, cmap='jet', alpha=0.5)
ax  = fig.add_subplot(1, 2, 2)
ax.contourf(bf_x1, bf_x2, bf_y, cmap='jet', alpha=0.5, levels=200)
ax.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], marker='x', color='w', s=100, linewidth=3.0)
ax.set_xlabel(r'$x_1$', size='xx-large')
ax.set_ylabel(r'$x_2$', size='xx-large', rotation='horizontal', labelpad=15)
ax.set_aspect('equal')
ax.set_title('contour plot of Branin function', size='xx-large')
plt.show()

# %%
kg = GPy.kern.RBF (input_dim=2, ARD = True)
kb = GPy.kern.Bias(input_dim=2)
mins_list = np.array([])
vars_list = np.array([])
th1_list  = np.array([])
th2_list  = np.array([])
th3_list  = np.array([])
th4_list  = np.array([])
ngrid = [5, 7, 10, 20]
#
for samp in range(3):
    for ntrn in ngrid:
        np.random.seed(seed)
        xg1   = np.linspace(-5, 10, ntrn)
        xg2   = np.linspace( 0, 15, ntrn)
        xg1   = np.linspace(1, 2 * ntrn - 1, ntrn) / (2.0 * ntrn) * 15.0 - 5.0
        xg2   = xg1 + 5.
        x1,x2 = np.meshgrid(xg1,xg2)
        if samp == 0:
            X = np.vstack((x1.flatten(),x2.flatten())).T
        if samp == 1:
            X = np.random.uniform(size=(ntrn ** 2, 2)) * 15.0
            X[:,0]= X[:,0] - 5.0
        if samp == 2:
            X = LHS(ntrn ** 2)
        Y     = branin(X)[:, None]
        # %%
        fig = plt.figure(figsize=(28.0, 7.0), dpi=100, tight_layout=False)
        gs  = plt.GridSpec(1,4)
        ax  = plt.subplot(gs.new_subplotspec((0, 0), 1, 1))
        map = ax.contourf(bf_x1, bf_x2, bf_y, cmap='jet', alpha=0.5, levels=100, vmin=0, vmax=300)
        ax.scatter(X[:,0], X[:,1], marker='o', s=30, color='white')
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(map, cax=cax)
        ax.set_aspect('equal')
        ax.set_title(r'Training Data Set ($n=%d$)' % ntrn ** 2, fontsize='x-large')
        
# %%
        k  = kg + kb
        m  = GPy.models.GPRegression(X, Y, k, normalizer=True)
        m.sum.rbf.variance.constrain_bounded     (1e-3, 1e6, warning=False)
        m.sum.bias.variance.constrain_bounded    (1e-3, 1e5, warning=False)
        m.sum.rbf.lengthscale.constrain_bounded  (1e-1, 1e3, warning=False)
        m.Gaussian_noise.variance.constrain_fixed(1e-6)
        m.optimize(max_iters=10000)
        th1_list = np.append(th1_list, m.sum.rbf.variance)
        th2_list = np.append(th2_list, 2 * m.sum.rbf.lengthscale[0] ** 2)
        th3_list = np.append(th3_list, 2 * m.sum.rbf.lengthscale[1] ** 2)
        th4_list = np.append(th4_list, m.sum.bias.variance)
# %%
        ax   = plt.subplot(gs.new_subplotspec((0, 1), 1, 1))
        cmap = plt.cm.get_cmap('jet').copy()
        cmap.set_under('silver')
        cmap.set_over ('dimgray')
        map = ax.contourf(bf_x1, bf_x2, 
                m.predict(bf_X)[0].reshape(151,151) - bf_y, cmap=cmap, alpha=0.5, levels=100, vmin=-5, vmax=5)
        xmin = np.array([[-np.pi,12.275],[np.pi,2.275],[9.42478,2.475]])
        mins = m.predict(xmin)
        mins_list = np.append(mins_list, mins[0][:])
        vars_list = np.append(vars_list, mins[1][:])
#
        ax.text(-2.2, 12.2,  r'%.4f$\pm$%.4f' %(np.squeeze(mins[0][0]), np.sqrt(np.squeeze(mins[1][0]))), 
                size='x-large', fontweight='bold')
        ax.text(-3.5, 1.5,  r'%.4f$\pm$%.4f' %(np.squeeze(mins[0][1]), np.sqrt(np.squeeze(mins[1][1]))), 
                size='x-large', fontweight='bold')
        ax.text(3.0, 3.0,  r'%.4f$\pm$%.4f' %(np.squeeze(mins[0][2]), np.sqrt(np.squeeze(mins[1][2]))), 
                size='x-large', fontweight='bold')
        ax.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], marker='x', color='k', s=50, linewidth=2.0)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(map, cax=cax)
        ax.set_aspect('equal')
        ax.set_title(r'${\cal L}(\bm{\upvartheta})_{\rm max}=%.4E$' %m.log_likelihood(), size='x-large')
# %%
        Xp      = np.random.uniform(size=(100000, 2)) * 15.0
        Xp[:,0] = Xp[:,0] - 5
        Yp      = m.predict(Xp)
        #
        Yp_mean = np.array([])
        Yp_prob = np.array([])
        Yp_pr_u = np.array([])
        Yp_pr_l = np.array([])
        Yp_sigm = np.array([])
        interval = np.linspace(1000, 100001, 100)
        for nn in interval:
            n = int(nn)
            sigm    = np.sqrt(np.mean(Yp[1][0:n]))
            Yp_mean = np.append(Yp_mean, np.mean(Yp[0][0:n]))
            Yp_prob = np.append(Yp_prob, np.mean(Yp[0][0:n] > 200))
            Yp_pr_u = np.append(Yp_pr_u, np.mean(Yp[0][0:n] > 200 - 2 * sigm))
            Yp_pr_l = np.append(Yp_pr_l, np.mean(Yp[0][0:n] > 200 + 2 * sigm))            
            Yp_sigm = np.append(Yp_sigm, sigm)
            
# %%
        ax = plt.subplot(gs.new_subplotspec((0, 2), 1, 1))
        ax.set_title(r'Pr$(f(\textbf{x}))>200$', fontsize='x-large')
        ax.plot(range(100), Yp_prob, linewidth=3, zorder=4)
        ax.fill_between(range(100), Yp_pr_l, Yp_pr_u, color='r', alpha=0.2)
        ax.plot([0, 99], [0.0123, 0.0123], linestyle='dashed', linewidth=2, color='r')
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xticklabels(np.arange(0, 100001, 20000))
        ax.set_xlim([0, 100])
        ax.set_box_aspect(0.9)
        #
        ax  = plt.subplot(gs.new_subplotspec((0, 3), 1, 1))
        ax.set_title(r'Expectation of $f(\textbf{x})$', fontsize='x-large')
        ax.plot(range(100), Yp_mean, linewidth=3, zorder=4) 
        ax.fill_between(range(100), Yp_mean + 2.0 * Yp_sigm, Yp_mean - 2.0 * Yp_sigm, color='r', alpha=0.2)
        ax.plot([0, 99], [54.31, 54.31], linestyle='dashed', linewidth=2, color='r')
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xticklabels(np.arange(0, 100001, 20000))
        ax.set_xlim([0, 100])
        ax.set_box_aspect(0.9)
        #
        plt.show()
        
# %%
lg  = ['square grid','Monte Carlo','Latin Hypercube']
mk  = ['x', 'o', 's']
cl  = ['blue', 'red', 'green']
fig = plt.figure(figsize=(12.0, 6.0), dpi=100, tight_layout=True)       
gs  = plt.GridSpec(2, 3)
#
ax  = plt.subplot(gs.new_subplotspec((0, 0), 1, 1))        
for smp in range(3):
    i = 4 * smp
    plt.plot(th1_list[i:i+4], th4_list[i:i+4], label=lg[smp], marker=mk[smp], color=cl[smp], alpha=0.5)
plt.loglog(base=10)
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
ax.set_xlabel(r'$\theta_1$', fontsize='large')
ax.set_ylabel(r'$\theta_4$', fontsize='large', rotation='horizontal')
plt.legend(loc='lower right', fontsize='xx-small')
plt.grid()
#
ax = plt.subplot(gs.new_subplotspec((1, 0), 1, 1))        
for smp in range(3):
    i = 4 * smp
    plt.plot(th2_list[i:i+4], th3_list[i:i+4], marker=mk[smp], label=lg[smp], color=cl[smp], alpha=0.5)
ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
ax.set_xlabel(r'$\theta_2$', fontsize='large')
ax.set_ylabel(r'$\theta_3$', fontsize='large', rotation='horizontal')
plt.legend(loc='lower right', fontsize='xx-small')
plt.grid()
##
ax = plt.subplot(gs.new_subplotspec((0, 1), 1, 2))        
k = 1
l = 0
for i in range(3):
    for j in range(4):
        for kk in range(3):
            ax.scatter(k, mins_list[l], marker=mk[i], color=cl[i])
            ax.plot([k, k],[mins_list[l] + np.sqrt(vars_list[l]), mins_list[l] - np.sqrt(vars_list[l])], color=cl[i])
            l += 1
            k += 1
        k += 1
    ax.text(8 + 16 * i, 8, lg[i], ha='center', fontsize='x-large')
ax.plot([0,48],[0.397887, 0.397887], linestyle='dotted')
ax.set_xticks(np.linspace(2,46,12))
ax.set_xticklabels(3 * [r'$n=25$', r'$n=49$', r'$n=100$', r'$n=400$'], rotation=30)
ax.set_xlim([0,48])
ylim = [-32,16]
ax.set_ylim(ylim)
for xx in range(4,48,4):
    ax.plot([xx, xx], ylim, color='grey', linewidth=1, linestyle='dotted')
#
ax = plt.subplot(gs.new_subplotspec((1, 1), 1, 2))        
k = 1
l = 0
for i in range(3):
    for j in range(4):
        for kk in range(3):
            ax.scatter(k, mins_list[l], marker=mk[i], color=cl[i])
            ax.plot([k, k],[mins_list[l] + np.sqrt(vars_list[l]), mins_list[l] - np.sqrt(vars_list[l])], color=cl[i])
            l += 1
            k += 1
        k += 1
ax.plot([0, 48],[0.397887, 0.397887], linestyle='dotted')
ax.set_xticks(np.linspace(2, 46, 12))
ax.set_xticklabels(3 * [r'$n=25$', r'$n=49$', r'$n=100$', r'$n=400$'], rotation=30)
ax.set_xlim([0, 48])
ylim=[0., 0.8]
ax.set_ylim(ylim)
for xx in range(4,48,4):
    ax.plot([xx, xx], ylim, color='grey', linewidth=1, linestyle='dotted')
plt.show()
