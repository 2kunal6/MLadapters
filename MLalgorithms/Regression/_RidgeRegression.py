
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.tol = tol
		self.max_iter = max_iter
		self.alpha = alpha
		self.solver = solver
		self.random_state = random_state
		Regression.__init__(self, copy_x=copy_x, normalize=normalize, fit_intercept=fit_intercept)
		self.model = Ridge(copy_x = self.copy_x,
			tol = self.tol,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			solver = self.solver,
			alpha = self.alpha,
			normalize = self.normalize,
			max_iter = self.max_iter)

