
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.random_state = random_state
		self.alpha = alpha
		self.selection = selection
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.normalize = normalize
		self.model = MLTR(copy_X = self.copy_X,
			normalize = self.normalize,
			max_iter = self.max_iter,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			tol = self.tol,
			random_state = self.random_state,
			alpha = self.alpha)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

