
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.max_iter = max_iter
		self.tol = tol
		self.normalize = normalize
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.alpha = alpha
		self.solver = solver
		self.copy_X = copy_X
		self.model = Ridge(max_iter = self.max_iter,
			alpha = self.alpha,
			normalize = self.normalize,
			solver = self.solver,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			random_state = self.random_state)

