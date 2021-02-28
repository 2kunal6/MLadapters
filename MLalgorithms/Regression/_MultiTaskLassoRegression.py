
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.selection = selection
		self.random_state = random_state
		self.max_iter = max_iter
		self.alpha = alpha
		self.tol = tol
		self.warm_start = warm_start
		Regression.__init__(self, normalize=normalize, copy_x=copy_x, fit_intercept=fit_intercept)
		self.model = MLTR(max_iter = self.max_iter,
			random_state = self.random_state,
			alpha = self.alpha,
			selection = self.selection,
			warm_start = self.warm_start,
			normalize = self.normalize,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_x = self.copy_x)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

