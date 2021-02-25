
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.max_iter = max_iter
		self.solver = solver
		self.random_state = random_state
		self.tol = tol
		self.alpha = alpha
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X)
		self.model = Ridge(fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			tol = self.tol,
			normalize = self.normalize,
			random_state = self.random_state,
			solver = self.solver,
			alpha = self.alpha,
			copy_X = self.copy_X)

