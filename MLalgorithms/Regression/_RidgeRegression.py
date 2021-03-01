
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.random_state = random_state
		self.solver = solver
		self.tol = tol
		self.max_iter = max_iter
		self.alpha = alpha
		Regression.__init__(self, fit_intercept=fit_intercept, copy_x=copy_x, normalize=normalize)
		self.model = Ridge(normalize = self.normalize,
			max_iter = self.max_iter,
			solver = self.solver,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			random_state = self.random_state,
			alpha = self.alpha,
			copy_x = self.copy_x)

