
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.random_state = random_state
		self.selection = selection
		self.positive = positive
		self.warm_start = warm_start
		self.tol = tol
		self.max_iter = max_iter
		self.alpha = alpha
		self.precompute = precompute
		Regression.__init__(self, normalize=normalize, copy_X=copy_X, fit_intercept=fit_intercept)
		self.model = Lasso(selection = self.selection,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			random_state = self.random_state,
			normalize = self.normalize,
			alpha = self.alpha,
			positive = self.positive,
			tol = self.tol,
			warm_start = self.warm_start,
			max_iter = self.max_iter)

