
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.tol = tol
		self.selection = selection
		self.normalize = normalize
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.alpha = alpha
		self.copy_X = copy_X
		self.model = MLTR(warm_start = self.warm_start,
			max_iter = self.max_iter,
			alpha = self.alpha,
			normalize = self.normalize,
			selection = self.selection,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			random_state = self.random_state)

