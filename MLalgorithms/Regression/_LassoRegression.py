
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.precompute = precompute
		self.selection = selection
		self.max_iter = max_iter
		self.positive = positive
		self.alpha = alpha
		self.warm_start = warm_start
		self.tol = tol
		self.random_state = random_state
		Regression.__init__(self, copy_x=copy_x, normalize=normalize, fit_intercept=fit_intercept)
		self.model = Lasso(copy_x = self.copy_x,
			tol = self.tol,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha,
			normalize = self.normalize,
			positive = self.positive,
			precompute = self.precompute,
			selection = self.selection,
			max_iter = self.max_iter)

