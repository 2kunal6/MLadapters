
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.precompute = precompute
		self.alpha = alpha
		self.max_iter = max_iter
		self.random_state = random_state
		self.warm_start = warm_start
		self.positive = positive
		self.selection = selection
		self.tol = tol
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X)
		self.model = Lasso(tol = self.tol,
			positive = self.positive,
			selection = self.selection,
			warm_start = self.warm_start,
			max_iter = self.max_iter,
			normalize = self.normalize,
			alpha = self.alpha,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			random_state = self.random_state)

