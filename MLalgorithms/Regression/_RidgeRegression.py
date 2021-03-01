
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.max_iter = max_iter
		self.tol = tol
		self.alpha = alpha
		self.solver = solver
		self.random_state = random_state
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_x=copy_x)
		self.model = Ridge(copy_x = self.copy_x,
			tol = self.tol,
			alpha = self.alpha,
			normalize = self.normalize,
			solver = self.solver,
			random_state = self.random_state,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept)

