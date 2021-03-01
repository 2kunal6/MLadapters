
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.random_state = random_state
		self.tol = tol
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.selection = selection
		self.normalize = normalize
		self.alpha = alpha
		self.model = MLTR(tol = self.tol,
			copy_X = self.copy_X,
			normalize = self.normalize,
			random_state = self.random_state,
			max_iter = self.max_iter,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

