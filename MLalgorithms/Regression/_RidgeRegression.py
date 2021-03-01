
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.copy_X = copy_X
		self.tol = tol
		self.normalize = normalize
		self.random_state = random_state
		self.alpha = alpha
		self.max_iter = max_iter
		self.solver = solver
		self.fit_intercept = fit_intercept
		self.model = Ridge(max_iter = self.max_iter,
			normalize = self.normalize,
			alpha = self.alpha,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			solver = self.solver,
			tol = self.tol,
			random_state = self.random_state)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

