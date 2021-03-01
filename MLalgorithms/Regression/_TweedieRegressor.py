
from sklearn.linear_model import TweedieRegressor as TR
from MLalgorithms._Regression import Regression


class TweedieRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0):
		self.alpha = alpha
		self.verbose = verbose
		self.tol = tol
		self.power = power
		self.max_iter = max_iter
		self.link = link
		self.fit_intercept = fit_intercept
		self.warm_start = warm_start
		self.model = TR(verbose = self.verbose,
			alpha = self.alpha,
			warm_start = self.warm_start,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			power = self.power,
			link = self.link)

