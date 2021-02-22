
from sklearn.linear_model import Lasso
from MLalgorithms.Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, fit_intercept, normalize, copy_X, positive = False, tol = 0.0001, random_state = None, max_iter = 1000, selection = 'cyclic', alpha = 1.0, precompute = False, warm_start = False):
		self.positive = positive
		self.tol = tol
		self.random_state = random_state
		self.max_iter = max_iter
		self.selection = selection
		self.alpha = alpha
		self.precompute = precompute
		self.warm_start = warm_start
		Regression.__init__(self, fit_intercept, normalize, copy_X)
		self.model = Lasso(max_iter = self.max_iter,
			warm_start = self.warm_start,
			fit_intercept = self.fit_intercept,
			selection = self.selection,
			positive = self.positive,
			copy_X = self.copy_X,
			random_state = self.random_state,
			normalize = self.normalize,
			alpha = self.alpha,
			tol = self.tol,
			precompute = self.precompute)

