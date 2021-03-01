
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.warm_start = warm_start
		self.tol = tol
		self.max_iter = max_iter
		self.random_state = random_state
		self.selection = selection
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = MLTR(alpha = self.alpha,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			normalize = self.normalize,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			selection = self.selection,
			max_iter = self.max_iter,
			tol = self.tol)

	def predict(self, X):
		return self.model.predict(X=X)

