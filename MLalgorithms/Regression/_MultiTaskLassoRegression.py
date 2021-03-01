
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.selection = selection
		self.random_state = random_state
		self.tol = tol
		self.alpha = alpha
		Regression.__init__(self, copy_x=copy_x, fit_intercept=fit_intercept, normalize=normalize)
		self.model = MLTR(selection = self.selection,
			max_iter = self.max_iter,
			copy_x = self.copy_x,
			normalize = self.normalize,
			random_state = self.random_state,
			warm_start = self.warm_start,
			fit_intercept = self.fit_intercept,
			alpha = self.alpha,
			tol = self.tol)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

