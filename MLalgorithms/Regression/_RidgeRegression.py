
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.solver = solver
		self.alpha = alpha
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.copy_X = copy_X
		self.normalize = normalize
		self.tol = tol
		self.random_state = random_state
		self.model = Ridge(normalize = self.normalize,
			alpha = self.alpha,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			random_state = self.random_state,
			copy_X = self.copy_X,
			solver = self.solver)

