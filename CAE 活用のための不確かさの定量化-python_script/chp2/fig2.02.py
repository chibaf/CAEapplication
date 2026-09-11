#===============================================================================
#    図　2.2 最小二乗法の例
#
#    Copyright (c) 2024, Yuukou TOYONORI
#    All rights reserved.
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
x = np.array([-5.0, -3.0, -1.0, 2.0, 5.0])
y = np.array([-5.3, -2.7, -0.7, 0.6, 4.9])

# %%
clf = linear_model.LinearRegression()
clf.fit(x.reshape(len(x),1), y.reshape(len(y),1))
a = clf.coef_[0][0]
b = clf.intercept_[0]

# %%
fig, ax = plt.subplots(dpi=100)
ax.plot(x, y, 'rx')
ax.plot([-6.0, 6.0], [-6.0 * a + b, 6.0 * a + b], 'b-')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$', rotation='horizontal')
ax.set_title('Least Square Method')
ax.set_xlim([-6.0, 6.0])
ax.set_ylim([-6.0, 6.0])
ax.set_aspect('equal')
ax.grid(linestyle='--', alpha=0.5)
plt.show()