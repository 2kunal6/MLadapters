
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.max_iter = max_iter
		self.selection = selection
		self.precompute = precompute
		self.alpha = alpha
		self.positive = positive
		self.warm_start = warm_start
		self.tol = tol
		self.random_state = random_state
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_x=copy_x)
		self.model = Lasso(warm_start = self.warm_start,
			copy_x = self.copy_x,
			tol = self.tol,
			precompute = self.precompute,
			alpha = self.alpha,
			selection = self.selection,
			normalize = self.normalize,
			positive = self.positive,
			random_state = self.random_state,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept)

