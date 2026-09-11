#===============================================================================
#   図　6.11  制約条件の例
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
x       = np.array(range(101), dtype=np.int16)
X, Y, Z = np.meshgrid(x, x, x)
X       = X.flatten()
Y       = Y.flatten()
Z       = Z.flatten()
# %%
c, XX, YY, ZZ = [], [], [], []
for i in range(len(X)):
    if X[i] + Y[i] + Z[i] == 100:
        c.append([X[i] / 100.0, Y[i] / 100.0, Z[i] / 100.0])
        XX.append(X[i])
        YY.append(Y[i])
        ZZ.append(Z[i])
# %%
fig = plt.figure(figsize=(10.0, 8.0), dpi=100)
ax  = fig.add_subplot(projection='3d')
ax.scatter(XX, YY, ZZ, c=c, alpha=0.5, s=5)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
ax.plot((100, 0, 0, 100), (0, 100, 0, 0), (0, 0, 100, 0), 'k')
ax.view_init(elev=45, azim=45)
ax.set_title(r'$x_1+x_2+x_3=100$', fontsize='xx-large')
ax.text2D(0.0, 0.0, r'blank', color='w', ha='center', fontsize='x-large', transform=ax.transAxes)
plt.show()

# %%
fig = plt.figure(figsize=(8.0, 8.0), dpi=100)
ax  = fig.add_subplot(projection='3d')
ax.scatter(XX, YY, ZZ, c=c, alpha=0.5, s=10)

ax.plot((100, 0, 0, 100), (0, 100, 0, 0), (0, 0, 100, 0), 'k', lw=0.5)
ax.view_init(elev=45, azim=45)
ax.set_xticks([]) 
ax.set_yticks([])
ax.set_zticks([])
ax.text2D(0.5, 0.97, r'${\bf x}=(0, 0, 100)$', ha='center', fontsize='x-large', transform=ax.transAxes)
ax.text2D(0.2, 0.3,  r'${\bf x}=(100, 0, 0)$', ha='center', fontsize='x-large', transform=ax.transAxes)
ax.text2D(0.9, 0.3,  r'${\bf x}=(0, 100, 0)$', ha='center', fontsize='x-large', transform=ax.transAxes)
plt.axis('off')
plt.show()
