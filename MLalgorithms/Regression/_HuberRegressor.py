
from sklearn.linear_model import HuberRegressor as HR
from MLalgorithms._Regression import Regression


class HuberRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, criterion='mse', max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
		self.warm_start = warm_start
		self.criterion = criterion
		self.tol = tol
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.model = HR(fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			max_iter = self.max_iter,
			alpha = self.alpha,
			criterion = self.criterion,
			tol = self.tol)

