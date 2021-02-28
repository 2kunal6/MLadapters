
from sklearn.linear_model import Lasso
from MLalgorithms._Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, precompute=False, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.positive = positive
		self.max_iter = max_iter
		self.selection = selection
		self.precompute = precompute
		self.warm_start = warm_start
		self.alpha = alpha
		self.random_state = random_state
		self.tol = tol
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_x=copy_x)
		self.model = Lasso(random_state = self.random_state,
			max_iter = self.max_iter,
			precompute = self.precompute,
			selection = self.selection,
			positive = self.positive,
			warm_start = self.warm_start,
			tol = self.tol,
			copy_x = self.copy_x,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize)

