
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.tol = tol
		self.normalize = normalize
		self.random_state = random_state
		self.alpha = alpha
		self.max_iter = max_iter
		self.selection = selection
		self.warm_start = warm_start
		self.fit_intercept = fit_intercept
		self.model = MLTR(max_iter = self.max_iter,
			warm_start = self.warm_start,
			normalize = self.normalize,
			alpha = self.alpha,
			copy_X = self.copy_X,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			random_state = self.random_state)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

