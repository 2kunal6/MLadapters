
from sklearn.linear_model import TweedieRegressor as TR
from MLalgorithms._Regression import Regression


class TweedieRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0):
		self.warm_start = warm_start
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.verbose = verbose
		self.tol = tol
		self.link = link
		self.alpha = alpha
		self.power = power
		self.model = TR(tol = self.tol,
			power = self.power,
			link = self.link,
			max_iter = self.max_iter,
			verbose = self.verbose,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

