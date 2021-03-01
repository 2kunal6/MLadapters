
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.random_state = random_state
		self.selection = selection
		self.tol = tol
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.alpha = alpha
		Regression.__init__(self, fit_intercept=fit_intercept, copy_x=copy_x, normalize=normalize)
		self.model = MLTR(normalize = self.normalize,
			max_iter = self.max_iter,
			tol = self.tol,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			copy_x = self.copy_x,
			random_state = self.random_state,
			alpha = self.alpha,
			warm_start = self.warm_start)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

