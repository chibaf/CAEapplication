import GPy
import numpy as np

X = np.array([ 0.00, 0.22, 0.44, 0.67, 0.90, 0.16, 0.34, 0.50, 0.73, 1.00])
Y = np.array([-0.06, 0.97, 0.25,-0.90,-0.53, 0.94, 0.85, 0.09,-0.93, 0.08])

model = GPy.models.GPRegression(X[:, None], Y[:, None])
print(model)
model.optimize()
print(model)
