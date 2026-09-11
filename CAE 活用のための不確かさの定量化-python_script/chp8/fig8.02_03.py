#===============================================================================
#   図　8.2  モンテカルロ法によるサンプリング
#   図　8.3  ラテン超方格法によるサンプリング
#
#   Copyright (c) 2024, Yuukou TOYONORI
#   All rights reserved.
#===============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pyDOE3 as doe
import pandas as pd
import seaborn as sns

plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['text.usetex']      = True

# %%
def LHS(dim, n, seed):
    lhs = doe.lhs(dim, samples=n, criterion='cm', iterations=10)
    return(lhs)

# %%
dim = 4
bin = 10
n   = bin ** 2

seed = 6789
np.random.seed(seed)
colors = [[np.random.rand(), np.random.rand(), np.random.rand()] for i in range(100)]
sns.set_theme(font_scale=1.5, style='ticks')

# %%
fig = plt.figure(figsize=(10.0, 10.0), dpi=100)
np.random.seed(seed)
mcs = np.random.random((n, dim))
df  = pd.DataFrame(mcs, columns=['X1','X2','X3','X4'])
sns.pairplot(df, diag_kws={'bins':bin}, plot_kws={'c':colors, 's':100})
plt.show()
#
fig = plt.figure(dpi=100)
lhs = LHS(dim, n, seed)
df  = pd.DataFrame(lhs, columns=['X1','X2','X3','X4'])
sns.pairplot(df, diag_kws={'bins':bin}, plot_kws={'c':colors, 's':100})
plt.show()

sns.reset_defaults()