#===============================================================================
#   図　3.9   最尤推定と最大事後確率推定（教師データ数：３）
#   図　3.10  最尤推定と最大事後確率推定（教師データ数：５）
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import multivariate_normal as mv_n

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
D = np.array([[-5.0, -5.4], [2.0, 0.5], [5.0, 4.8], [-1.0, -0.8], [-3.0, -3.1], [3.0, 2.8]])

# %%
def Draw(ndata, samples):
    plt.figure(figsize=(8.0, 8.0), dpi=100, tight_layout=True)
    gs  = GridSpec(2, 2) 
    #
    x, y = D[0:ndata, 0], D[0:ndata, 1]
    X    = np.stack((np.ones(len(x)), x))
    ys   = np.mat(y).T   
    Xd   = np.mat(X).T   
    xt   = np.linspace(-6.0, 6.0, 101)
    Xt   = np.mat(np.stack((np.ones(len(xt)), xt))).T
 
    # -----------------------------------------------------------------------------
    #   (1)最小二乗法によって重みベクトルの最尤推定値 w_MLを求める
    # -----------------------------------------------------------------------------
    W_ML  = (Xd.T * Xd).I * Xd.T * ys
    y_hat = np.array(Xt * W_ML)
    w_ML  = np.array(W_ML).flatten()
    # -----------------------------------------------------------------------------
    ax1 = plt.subplot(gs.new_subplotspec((0, 0), 1, 1))
    ax1.scatter(np.array(x), np.array(y), marker='x', color='red', zorder=3)
    ax1.plot(xt, y_hat, color='blue')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$', rotation='horizontal')
    ax1.set_xlim([-6.0, 6.0])
    ax1.set_ylim([-6.0, 6.0])
    ax1.set_aspect('equal')
    ax1.grid(linestyle='dotted')
    ax1.set_title('Least square method')
    # -----------------------------------------------------------------------------
    #   (2)尤度 𝑝(𝐲|X,𝐰) を求める
    # -----------------------------------------------------------------------------
    w      = np.linspace(-2, 2, 101)
    W0, W1 = np.meshgrid(w, w)
    w0     = np.array(W0).flatten()
    w1     = np.array(W1).flatten() 
    wg     = np.mat([w0, w1])
    y_wg   = (Xd * wg).T
    lik    = mv_n.pdf(y_wg, y, np.mat(np.identity(len(ys))))
    lik    = lik.reshape(np.array([len(w), len(w)])) 
    # -----------------------------------------------------------------------------
    ax2 = plt.subplot(gs.new_subplotspec((0, 1), 1, 1))
    plt.contourf(w, w, lik, levels=20)
    ax2.scatter(w_ML[0], w_ML[1], marker='*', label='MLE')
    ax2.set_xlim([-2.0, 1.0])
    ax2.set_ylim([-1.0, 2.0])
    ax2.set_xlabel('$w_0$')
    ax2.set_ylabel('$w_1$', rotation='horizontal')
    ax2.set_aspect('equal')
    ax2.grid(linestyle='dotted')
    ax2.set_title(r'Likelyhood $p\,(\mathbf{y}\,|\,\mathrm{X},\mathbf{w})$')
    # -----------------------------------------------------------------------------
    #   (3)事前分布を 𝑝(𝐰)=𝒩(𝟎,I) としたとき、事後分布 𝑝(𝐰|𝐲,X) を求める
    # -----------------------------------------------------------------------------
    var   = 1.
    var_2 = var ** -2
    cov   = np.mat(np.identity(2))
    w_mu  = var_2 * (var_2 * Xd.T * Xd + cov.I).I * Xd.T * ys
    A     = var_2 * Xd.T * Xd + cov.I
    # -----------------------------------------------------------------------------
    WW     = np.dstack((W0, W1))
    mu     = np.array(w_mu).flatten()
    Sigma  = A.I
    w_post = mv_n.pdf(WW, mu, Sigma)
    
    # -----------------------------------------------------------------------------
    #   (4)事後分布のパラメータ空間での等高線を描く
    # -----------------------------------------------------------------------------
    Sx  = np.sqrt(Sigma[0, 0])
    Sy  = np.sqrt(Sigma[1, 1])
    rho = Sigma[0, 1] / Sx / Sy
    K   = np.array([2.0, 1.0])
    L   = np.exp(- K * K / 2.0) / (2.0 * np.pi * Sx * Sy * np.sqrt(1 - rho ** 2))
    # -----------------------------------------------------------------------------
    ax4 = plt.subplot(gs.new_subplotspec((1, 1), 1, 1))
    ax4.scatter(mu[0], mu[1],     marker='o', color='red',  label='MAP')
    ax4.scatter(w_ML[0], w_ML[1], marker='*', color='blue', label='MLE')
    ax4.set_xlim([-2.0, 1.0])
    ax4.set_ylim([-1.0, 2.0])
    ax4.set_xlabel('$w_0$')
    ax4.set_ylabel('$w_1$', rotation='horizontal')
    ax4.set_aspect('equal')
    ax4.grid(linestyle='dotted')
    ax4.set_title(r'Posterior distribution $p\,(\mathbf{w}\,|\,\mathrm{X},\mathbf{y})$')
    plt.contour(w ,w, w_post, levels=[L[0]], colors='grey',  linestyles='dotted')
    plt.contour(w ,w, w_post, levels=[L[1]], colors='grey',  linestyles='dotted')    
    # -----------------------------------------------------------------------------
    #   参考：共分散行列の幾何学的な意味（固有値分解） 
    # -----------------------------------------------------------------------------
    Eig, U = np.linalg.eig(Sigma)
    for i in range(2):
        v = np.array(np.sqrt(Eig[i]) * U[:, i]).flatten()
        ax4.annotate('', xy=[mu[0] + v[0], mu[1] + v[1]], xytext=[mu[0], mu[1]], 
                     arrowprops=dict(facecolor='green', edgecolor='green', width=0.5, headwidth=4, headlength=4))
    ax4.plot([mu[0], mu[0]], [mu[1], mu[1]], color='green', label='eigen vector') 
    ax4.text(mu[0] - v[0],       mu[1] - v[1],       r'$\Delta=1$', va='top', color='red', size='small')
    ax4.text(mu[0] - 2.0 * v[0], mu[1] - 2.0 * v[1], r'$\Delta=2$', va='top', color='red', size='small')
    ax4.legend(loc='lower right')
        
    # -----------------------------------------------------------------------------
    #   (5)テスト入力に対する出力の予測をデータ空間で描く
    # -----------------------------------------------------------------------------
    y_mu  = np.array(var_2 * Xt * Sigma * Xd.T * ys).flatten()
    y_sig = np.sqrt(np.diag(Xt * Sigma * Xt.T))
    # -----------------------------------------------------------------------------
    ax3 = plt.subplot(gs.new_subplotspec((1, 0), 1, 1))
    ax3.scatter(np.array(x), np.array(y), marker='x', color='blue', zorder=3)
    ax3.plot(xt, y_mu, color='red', label=r'$\mu$')
    ax3.fill_between(xt, y_mu - 2.0 * y_sig, y_mu + 2.0 * y_sig, facecolor='lightgray', alpha=0.5, label=r'$\mu\pm2\sigma$')
    ax3.set_xlabel('$x^*$')
    ax3.set_ylabel('$y^*$', rotation='horizontal')
    ax3.set_xlim([-6.0, 6.0])
    ax3.set_ylim([-6.0, 6.0])
    ax3.set_aspect('equal')
    ax3.grid(linestyle='dotted')
    ax3.set_title('Predictive distribution (MAP)')
    ##
    colors = ['magenta', 'cyan', 'brown', 'red', 'green', 'blue', 'yellow', 'gold', 'lime', 'plum']
    np.random.seed(1234)
    for s in range(samples):
        sample = np.random.multivariate_normal(mu, Sigma)
        ax3.plot(xt, sample[0] + sample[1] * xt, linewidth=1, linestyle='dashdot', color=colors[s])
        ax4.scatter(sample[0], sample[1], marker='+', color=colors[s], s=50)
    ax3.legend(loc='lower right')
    plt.show()
    
# %%
Draw(ndata=2, samples=0)
Draw(ndata=3, samples=0)
Draw(ndata=4, samples=0)
Draw(ndata=5, samples=0)
Draw(ndata=5, samples=5)
