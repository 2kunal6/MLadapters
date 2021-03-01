
from sklearn.linear_model import HuberRegressor as HR
from MLalgorithms._Regression import Regression


class HuberRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, criterion='mse', max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05):
		self.criterion = criterion
		self.warm_start = warm_start
		self.alpha = alpha
		self.max_iter = max_iter
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.model = HR(tol = self.tol,
			criterion = self.criterion,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

