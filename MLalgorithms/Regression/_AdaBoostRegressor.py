
from sklearn.ensemble import AdaBoostRegressor as ABR
from MLalgorithms._Regression import Regression


class AdaBoostRegressor(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, random_state=None):
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.random_state = random_state
		self.base_estimator = base_estimator
		self.model = ABR(base_estimator = self.base_estimator,
			learning_rate = self.learning_rate,
			random_state = self.random_state,
			n_estimators = self.n_estimators)

