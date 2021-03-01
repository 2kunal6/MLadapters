
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostRegressor as ABR
from MLalgorithms._Regression import Regression


class AdaBoostRegressor(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=None, fit_intercept=True, normalize=False, copy_X=True):
		self.random_state = random_state
		self.learning_rate = learning_rate
		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = ABR(n_estimators = self.n_estimators,
			copy_X = self.copy_X,
			normalize = self.normalize,
			learning_rate = self.learning_rate,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			base_estimator = self.base_estimator)

	def predict(self, X):
		return self.model.predict(X=X)

