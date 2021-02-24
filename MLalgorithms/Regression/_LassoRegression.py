
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.positive = positive
		self.alpha = alpha
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.selection = selection
		self.random_state = random_state
		self.precompute = precompute
		self.tol = tol
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = Lasso(warm_start = self.warm_start,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			positive = self.positive,
			copy_X = self.copy_X,
			normalize = self.normalize,
			max_iter = self.max_iter,
			precompute = self.precompute,
			selection = self.selection,
			random_state = self.random_state,
			alpha = self.alpha)

