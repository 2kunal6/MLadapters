
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.random_state = random_state
		self.tol = tol
		self.positive = positive
		self.selection = selection
		self.warm_start = warm_start
		self.alpha = alpha
		self.precompute = precompute
		self.max_iter = max_iter
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X)
		self.model = Lasso(fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			positive = self.positive,
			tol = self.tol,
			random_state = self.random_state,
			warm_start = self.warm_start,
			normalize = self.normalize,
			selection = self.selection,
			precompute = self.precompute,
			alpha = self.alpha,
			copy_X = self.copy_X)

