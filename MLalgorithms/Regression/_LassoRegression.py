
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.positive = positive
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self.selection = selection
		self.precompute = precompute
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = Lasso(alpha = self.alpha,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			normalize = self.normalize,
			positive = self.positive,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			selection = self.selection,
			max_iter = self.max_iter,
			tol = self.tol)

