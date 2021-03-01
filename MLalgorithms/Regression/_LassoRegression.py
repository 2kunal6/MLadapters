
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.warm_start = warm_start
		self.precompute = precompute
		self.normalize = normalize
		self.positive = positive
		self.random_state = random_state
		self.alpha = alpha
		self.selection = selection
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.max_iter = max_iter
		self.model = Lasso(copy_X = self.copy_X,
			normalize = self.normalize,
			precompute = self.precompute,
			max_iter = self.max_iter,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			tol = self.tol,
			random_state = self.random_state,
			alpha = self.alpha,
			positive = self.positive)

