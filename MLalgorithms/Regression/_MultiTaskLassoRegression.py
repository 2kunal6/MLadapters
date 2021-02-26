
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.random_state = random_state
		self.max_iter = max_iter
		self.selection = selection
		self.warm_start = warm_start
		self.tol = tol
		self.alpha = alpha
		Regression.__init__(self, normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
		self.model = MLTR(normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			alpha = self.alpha,
			warm_start = self.warm_start,
			max_iter = self.max_iter,
			selection = self.selection,
			tol = self.tol,
			random_state = self.random_state)

