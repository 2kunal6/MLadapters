
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.precompute = precompute
		self.random_state = random_state
		self.tol = tol
		self.positive = positive
		self.selection = selection
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.alpha = alpha
		Regression.__init__(self, fit_intercept=fit_intercept, copy_x=copy_x, normalize=normalize)
		self.model = Lasso(normalize = self.normalize,
			max_iter = self.max_iter,
			tol = self.tol,
			positive = self.positive,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			copy_x = self.copy_x,
			precompute = self.precompute,
			alpha = self.alpha,
			random_state = self.random_state,
			warm_start = self.warm_start)

