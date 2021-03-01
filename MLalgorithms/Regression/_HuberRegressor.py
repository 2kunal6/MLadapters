
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor as HR
from MLalgorithms._Regression import Regression


class HuberRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, criterion='mse', max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05, normalize=False, copy_X=True):
		self.warm_start = warm_start
		self.criterion = criterion
		self.max_iter = max_iter
		self.alpha = alpha
		self.tol = tol
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = HR(alpha = self.alpha,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			tol = self.tol,
			criterion = self.criterion)

	def predict(self, X):
		return self.model.predict(X=X)

