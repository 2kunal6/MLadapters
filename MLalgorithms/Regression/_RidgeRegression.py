
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.random_state = random_state
		self.alpha = alpha
		self.max_iter = max_iter
		self.solver = solver
		self.tol = tol
		Regression.__init__(self, copy_X=copy_X, normalize=normalize, fit_intercept=fit_intercept)
		self.model = Ridge(random_state = self.random_state,
			copy_X = self.copy_X,
			alpha = self.alpha,
			normalize = self.normalize,
			tol = self.tol,
			max_iter = self.max_iter,
			solver = self.solver,
			fit_intercept = self.fit_intercept)

