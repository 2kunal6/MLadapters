
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.copy_X = copy_X
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.selection = selection
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.random_state = random_state
		self.normalize = normalize
		self.model = MLTR(selection = self.selection,
			random_state = self.random_state,
			tol = self.tol,
			normalize = self.normalize,
			alpha = self.alpha,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

