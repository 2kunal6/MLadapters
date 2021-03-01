
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.max_iter = max_iter
		self.random_state = random_state
		self.tol = tol
		self.alpha = alpha
		self.solver = solver
		Regression.__init__(self, copy_x=copy_x, fit_intercept=fit_intercept, normalize=normalize)
		self.model = Ridge(max_iter = self.max_iter,
			copy_x = self.copy_x,
			normalize = self.normalize,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			alpha = self.alpha,
			solver = self.solver,
			tol = self.tol)

