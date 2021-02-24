
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.alpha = alpha
		self.solver = solver
		self.tol = tol
		self.max_iter = max_iter
		self.random_state = random_state
		Regression.__init__(self, normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
		self.model = Ridge(tol = self.tol,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			normalize = self.normalize,
			alpha = self.alpha,
			solver = self.solver,
			random_state = self.random_state)

