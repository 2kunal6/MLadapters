
from sklearn.linear_model import Ridge
from MLalgorithms.Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, fit_intercept, normalize, copy_X, tol = 0.001, random_state = None, solver = 'auto', alpha = 1.0, max_iter = None):
		self.tol = tol
		self.random_state = random_state
		self.solver = solver
		self.alpha = alpha
		self.max_iter = max_iter
		Regression.__init__(self, fit_intercept, normalize, copy_X)
		self.model = Ridge(max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			random_state = self.random_state,
			normalize = self.normalize,
			alpha = self.alpha,
			solver = self.solver,
			tol = self.tol)

