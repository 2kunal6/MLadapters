
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.copy_X = copy_X
		self.random_state = random_state
		self.tol = tol
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.solver = solver
		self.normalize = normalize
		self.alpha = alpha
		self.model = Ridge(tol = self.tol,
			copy_X = self.copy_X,
			normalize = self.normalize,
			random_state = self.random_state,
			max_iter = self.max_iter,
			solver = self.solver,
			fit_intercept = self.fit_intercept,
			alpha = self.alpha)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

