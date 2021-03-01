
from sklearn.linear_model import TweedieRegressor as TR
from MLalgorithms._Regression import Regression


class TweedieRegressor(Regression):
	
	def __init__(self, power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0):
		self.link = link
		self.verbose = verbose
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.power = power
		self.alpha = alpha
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.model = TR(link = self.link,
			power = self.power,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			verbose = self.verbose,
			warm_start = self.warm_start,
			tol = self.tol,
			alpha = self.alpha)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

