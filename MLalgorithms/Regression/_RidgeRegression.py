
from sklearn.linear_model import Ridge
from MLalgorithms._Regression import Regression


class RidgeRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
		self.tol = tol
		self.copy_X = copy_X
		self.max_iter = max_iter
		self.solver = solver
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.random_state = random_state
		self.normalize = normalize
		self.model = Ridge(random_state = self.random_state,
			tol = self.tol,
			normalize = self.normalize,
			alpha = self.alpha,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			solver = self.solver)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

