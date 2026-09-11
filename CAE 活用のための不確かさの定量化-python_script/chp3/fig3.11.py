#===============================================================================
#   図　3.11 累積分布関数と誤差関数
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 'xx-large'

# %%
x   = np.arange(-4.0, 4.01, 0.01)
erf = special.erf(x / np.sqrt(2.0))

# %%
fig, ax = plt.subplots(dpi=100)
ax.plot(x, (1.0 + erf) / 2.0, 'b-',  label=r'$\Phi (x)$')
ax.plot(x, erf, 'r:', linewidth=2.5, label=r'erf($x$)')
ax.set_xlabel('$x$')
ax.set_title('CDF and error function')
ax.grid(linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize='large')
plt.show()
