
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.selection = selection
		self.normalize = normalize
		self.precompute = precompute
		self.random_state = random_state
		self.positive = positive
		self.fit_intercept = fit_intercept
		self.alpha = alpha
		self.copy_X = copy_X
		self.model = Lasso(warm_start = self.warm_start,
			max_iter = self.max_iter,
			alpha = self.alpha,
			precompute = self.precompute,
			normalize = self.normalize,
			positive = self.positive,
			selection = self.selection,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			random_state = self.random_state)

