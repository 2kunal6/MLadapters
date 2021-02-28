
from sklearn.linear_model import MultiTaskLasso as MLTR
from MLalgorithms._Regression import Regression


class MultiTaskLassoRegression(Regression):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, warm_start=False, random_state=None, selection='cyclic'):
		self.selection = selection
		self.tol = tol
		self.max_iter = max_iter
		self.alpha = alpha
		self.warm_start = warm_start
		self.random_state = random_state
		Regression.__init__(self, copy_x=copy_x, normalize=normalize, fit_intercept=fit_intercept)
		self.model = MLTR(copy_x = self.copy_x,
			tol = self.tol,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha,
			normalize = self.normalize,
			max_iter = self.max_iter,
			selection = self.selection)

