
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.solver = solver
		self.tol = tol
		self.alpha = alpha
		self.max_iter = max_iter
		self.random_state = random_state
		Regression.__init__(self, copy_X=copy_X, normalize=normalize, fit_intercept=fit_intercept)
		self.model = Ridge(solver = self.solver,
			copy_X = self.copy_X,
			alpha = self.alpha,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			max_iter = self.max_iter,
			tol = self.tol)

