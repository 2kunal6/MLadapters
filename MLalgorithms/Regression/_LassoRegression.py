
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.warm_start = warm_start
		self.alpha = alpha
		self.tol = tol
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.positive = positive
		self.copy_X = copy_X
		self.normalize = normalize
		self.precompute = precompute
		self.random_state = random_state
		self.selection = selection
		self.model = Lasso(precompute = self.precompute,
			selection = self.selection,
			normalize = self.normalize,
			warm_start = self.warm_start,
			alpha = self.alpha,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			random_state = self.random_state,
			copy_X = self.copy_X,
			positive = self.positive)

