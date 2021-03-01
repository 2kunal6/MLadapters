
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.random_state = random_state
		self.warm_start = warm_start
		self.fit_intercept = fit_intercept
		self.positive = positive
		self.max_iter = max_iter
		self.tol = tol
		self.selection = selection
		self.normalize = normalize
		self.alpha = alpha
		self.precompute = precompute
		self.model = Lasso(tol = self.tol,
			precompute = self.precompute,
			copy_X = self.copy_X,
			normalize = self.normalize,
			random_state = self.random_state,
			positive = self.positive,
			max_iter = self.max_iter,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha)

