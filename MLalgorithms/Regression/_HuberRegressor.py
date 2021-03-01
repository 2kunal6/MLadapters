
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor as HR
from MLalgorithms._Regression import Regression


class HuberRegressor(Regression):
	
	def __init__(self, criterion='mse', max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
		self.max_iter = max_iter
		self.criterion = criterion
		self.alpha = alpha
		self.warm_start = warm_start
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.model = HR(max_iter = self.max_iter,
			warm_start = self.warm_start,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			criterion = self.criterion)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

