import numpy as np
import GPy
import matplotlib.pyplot as plt

X = np.array([ 0.00, 0.22, 0.44, 0.67, 0.90, 0.16, 0.34, 0.50, 0.73, 1.00])
Y = np.array([-0.06, 0.97, 0.25,-0.90,-0.53, 0.94, 0.85, 0.09,-0.93, 0.08])

model = GPy.models.GPRegression(X[:, None], Y[:, None])
model.optimize()
model.plot()
##
num_sample = 5
xtest      = np.linspace(-0.2, 1.2, 201)[:, None]
mean, cov  = model.predict_noiseless(xtest, full_cov=True)
mean       = mean.flatten()

sample = np.random.multivariate_normal(mean, cov, size=num_sample)
for i in range(num_sample):    
    plt.plot(xtest, sample[i, :], linewidth=2.0, linestyle='dotted')
    