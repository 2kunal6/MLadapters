
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.warm_start = warm_start
		self.random_state = random_state
		self.max_iter = max_iter
		self.selection = selection
		self.positive = positive
		self.alpha = alpha
		self.precompute = precompute
		Regression.__init__(self, copy_X=copy_X, normalize=normalize, fit_intercept=fit_intercept)
		self.model = Lasso(warm_start = self.warm_start,
			max_iter = self.max_iter,
			random_state = self.random_state,
			tol = self.tol,
			copy_X = self.copy_X,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			positive = self.positive,
			selection = self.selection,
			alpha = self.alpha,
			precompute = self.precompute)

