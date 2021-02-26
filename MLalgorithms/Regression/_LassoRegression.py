
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.selection = selection
		self.max_iter = max_iter
		self.positive = positive
		self.tol = tol
		self.warm_start = warm_start
		self.random_state = random_state
		self.alpha = alpha
		self.precompute = precompute
		Regression.__init__(self, normalize=normalize, copy_X=copy_X, fit_intercept=fit_intercept)
		self.model = Lasso(copy_X = self.copy_X,
			alpha = self.alpha,
			warm_start = self.warm_start,
			normalize = self.normalize,
			tol = self.tol,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			positive = self.positive,
			precompute = self.precompute,
			random_state = self.random_state,
			max_iter = self.max_iter)

