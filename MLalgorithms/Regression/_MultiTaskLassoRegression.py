
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.random_state = random_state
		self.max_iter = max_iter
		self.selection = selection
		self.fit_intercept = fit_intercept
		self.copy_X = copy_X
		self.tol = tol
		self.alpha = alpha
		self.normalize = normalize
		self.warm_start = warm_start
		self.model = MLTR(normalize = self.normalize,
			copy_X = self.copy_X,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			selection = self.selection,
			warm_start = self.warm_start,
			tol = self.tol,
			alpha = self.alpha)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

