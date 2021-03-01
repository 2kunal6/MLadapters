
from sklearn.linear_model import TweedieRegressor as TR
from MLalgorithms._Regression import Regression


class TweedieRegressor(Regression):
	
	def __init__(self, power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0):
		self.power = power
		self.warm_start = warm_start
		self.link = link
		self.verbose = verbose
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.max_iter = max_iter
		self.model = TR(power = self.power,
			link = self.link,
			verbose = self.verbose,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			tol = self.tol,
			alpha = self.alpha)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

