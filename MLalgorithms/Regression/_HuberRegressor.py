
from sklearn.linear_model import HuberRegressor as HR
from MLalgorithms._Regression import Regression


class HuberRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, criterion='mse', max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
		self.criterion = criterion
		self.max_iter = max_iter
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.warm_start = warm_start
		self.model = HR(warm_start = self.warm_start,
			alpha = self.alpha,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			criterion = self.criterion)

