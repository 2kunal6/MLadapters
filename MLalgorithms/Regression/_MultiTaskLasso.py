
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLasso(Regression):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.alpha = alpha
		self.random_state = random_state
		self.selection = selection
		Regression.__init__(self, copy_X=copy_X, fit_intercept=fit_intercept, normalize=normalize)
		self.model = MLTR(selection = self.selection,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			max_iter = self.max_iter,
			copy_X = self.copy_X,
			warm_start = self.warm_start,
			random_state = self.random_state,
			tol = self.tol)

	def predict(self, X):
		return self.model.predict(X=X)

