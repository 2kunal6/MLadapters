
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.positive = positive
		self.precompute = precompute
		self.selection = selection
		self.tol = tol
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.random_state = random_state
		Regression.__init__(self, normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
		self.model = Lasso(precompute = self.precompute,
			tol = self.tol,
			selection = self.selection,
			copy_X = self.copy_X,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			warm_start = self.warm_start,
			normalize = self.normalize,
			alpha = self.alpha,
			positive = self.positive)

