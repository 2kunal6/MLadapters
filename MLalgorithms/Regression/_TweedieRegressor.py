
from sklearn.linear_model import TweedieRegressor as TR
from MLalgorithms._Regression import Regression


class TweedieRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, power=0.0, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0):
		self.alpha = alpha
		self.warm_start = warm_start
		self.verbose = verbose
		self.power = power
		self.max_iter = max_iter
		self.tol = tol
		self.link = link
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = TR(tol = self.tol,
			link = self.link,
			alpha = self.alpha,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			verbose = self.verbose,
			max_iter = self.max_iter,
			power = self.power)

	def predict(self, X):
		return self.model.predict(X=X)

