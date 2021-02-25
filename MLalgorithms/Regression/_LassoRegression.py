
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.precompute = precompute
		self.selection = selection
		self.alpha = alpha
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.random_state = random_state
		self.positive = positive
		Regression.__init__(self, copy_X=copy_X, normalize=normalize, fit_intercept=fit_intercept)
		self.model = Lasso(tol = self.tol,
			selection = self.selection,
			copy_X = self.copy_X,
			positive = self.positive,
			alpha = self.alpha,
			random_state = self.random_state,
			warm_start = self.warm_start,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			max_iter = self.max_iter,
			precompute = self.precompute)

