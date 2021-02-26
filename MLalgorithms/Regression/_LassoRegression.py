
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.max_iter = max_iter
		self.precompute = precompute
		self.random_state = random_state
		self.selection = selection
		self.warm_start = warm_start
		self.positive = positive
		self.tol = tol
		self.alpha = alpha
		Regression.__init__(self, normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
		self.model = Lasso(normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			positive = self.positive,
			alpha = self.alpha,
			warm_start = self.warm_start,
			max_iter = self.max_iter,
			precompute = self.precompute,
			selection = self.selection,
			tol = self.tol,
			random_state = self.random_state)

