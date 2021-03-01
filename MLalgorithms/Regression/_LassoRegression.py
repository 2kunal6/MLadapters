
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.precompute = precompute
		self.random_state = random_state
		self.selection = selection
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.tol = tol
		self.copy_X = copy_X
		self.alpha = alpha
		self.positive = positive
		self.normalize = normalize
		self.warm_start = warm_start
		self.model = Lasso(normalize = self.normalize,
			copy_X = self.copy_X,
			random_state = self.random_state,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			selection = self.selection,
			warm_start = self.warm_start,
			positive = self.positive,
			tol = self.tol,
			alpha = self.alpha)

