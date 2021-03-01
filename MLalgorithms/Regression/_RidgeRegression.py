
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.random_state = random_state
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.solver = solver
		self.copy_X = copy_X
		self.tol = tol
		self.alpha = alpha
		self.normalize = normalize
		self.model = Ridge(normalize = self.normalize,
			copy_X = self.copy_X,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			solver = self.solver,
			tol = self.tol,
			alpha = self.alpha)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

