
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.warm_start = warm_start
		self.precompute = precompute
		self.selection = selection
		self.random_state = random_state
		self.positive = positive
		self.alpha = alpha
		self.max_iter = max_iter
		Regression.__init__(self, copy_x=copy_x, fit_intercept=fit_intercept, normalize=normalize)
		self.model = Lasso(selection = self.selection,
			max_iter = self.max_iter,
			copy_x = self.copy_x,
			normalize = self.normalize,
			random_state = self.random_state,
			warm_start = self.warm_start,
			fit_intercept = self.fit_intercept,
			alpha = self.alpha,
			positive = self.positive,
			precompute = self.precompute,
			tol = self.tol)

