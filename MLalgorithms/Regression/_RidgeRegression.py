
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, fit_intercept = True, normalize = False, copy_X = True, max_iter = None, random_state = None, alpha = 1.0, solver = 'auto', tol = 0.001):
		self.max_iter = max_iter
		self.random_state = random_state
		self.alpha = alpha
		self.solver = solver
		self.tol = tol
		Regression.__init__(self, fit_intercept, normalize, copy_X)
		self.model = Ridge(random_state = self.random_state,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			max_iter = self.max_iter,
			normalize = self.normalize,
			solver = self.solver,
			alpha = self.alpha)

