
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.normalize = normalize
		self.tol = tol
		self.copy_X = copy_X
		self.selection = selection
		self.warm_start = warm_start
		self.alpha = alpha
		self.random_state = random_state
		self.positive = positive
		self.precompute = precompute
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.model = Lasso(copy_X = self.copy_X,
			positive = self.positive,
			max_iter = self.max_iter,
			random_state = self.random_state,
			alpha = self.alpha,
			precompute = self.precompute,
			tol = self.tol,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			selection = self.selection,
			warm_start = self.warm_start)

