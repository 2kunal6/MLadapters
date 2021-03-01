
from sklearn.linear_model import TweedieRegressor as TR
from MLalgorithms._Regression import Regression


class TweedieRegressor(Regression):
	
	def __init__(self, power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0):
		self.power = power
		self.max_iter = max_iter
		self.verbose = verbose
		self.link = link
		self.alpha = alpha
		self.tol = tol
		self.warm_start = warm_start
		self.fit_intercept = fit_intercept
		self.model = TR(max_iter = self.max_iter,
			warm_start = self.warm_start,
			power = self.power,
			alpha = self.alpha,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			verbose = self.verbose,
			link = self.link)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

