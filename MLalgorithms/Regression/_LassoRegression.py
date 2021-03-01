
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.warm_start = warm_start
		self.selection = selection
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.precompute = precompute
		self.tol = tol
		self.random_state = random_state
		self.max_iter = max_iter
		self.normalize = normalize
		self.positive = positive
		self.model = Lasso(selection = self.selection,
			random_state = self.random_state,
			tol = self.tol,
			normalize = self.normalize,
			alpha = self.alpha,
			positive = self.positive,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			max_iter = self.max_iter)

