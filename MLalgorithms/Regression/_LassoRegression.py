
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, fit_intercept = True, normalize = False, copy_X = True, positive = False, precompute = False, warm_start = False, max_iter = 1000, random_state = None, alpha = 1.0, tol = 0.0001, selection = 'cyclic'):
		self.positive = positive
		self.precompute = precompute
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.random_state = random_state
		self.alpha = alpha
		self.tol = tol
		self.selection = selection
		Regression.__init__(self, fit_intercept, normalize, copy_X)
		self.model = Lasso(random_state = self.random_state,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			max_iter = self.max_iter,
			precompute = self.precompute,
			normalize = self.normalize,
			selection = self.selection,
			positive = self.positive,
			alpha = self.alpha,
			warm_start = self.warm_start)

