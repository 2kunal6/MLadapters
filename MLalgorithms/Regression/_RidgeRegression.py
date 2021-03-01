
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.copy_X = copy_X
		self.solver = solver
		self.max_iter = max_iter
		self.random_state = random_state
		self.alpha = alpha
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.model = Ridge(copy_X = self.copy_X,
			normalize = self.normalize,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			random_state = self.random_state,
			alpha = self.alpha,
			solver = self.solver)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

