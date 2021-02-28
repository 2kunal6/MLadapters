
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.selection = selection
		self.tol = tol
		self.random_state = random_state
		self.precompute = precompute
		self.positive = positive
		self.alpha = alpha
		self.max_iter = max_iter
		self.warm_start = warm_start
		Regression.__init__(self, normalize=normalize, copy_x=copy_x, fit_intercept=fit_intercept)
		self.model = Lasso(max_iter = self.max_iter,
			precompute = self.precompute,
			positive = self.positive,
			random_state = self.random_state,
			alpha = self.alpha,
			selection = self.selection,
			warm_start = self.warm_start,
			normalize = self.normalize,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_x = self.copy_x)

