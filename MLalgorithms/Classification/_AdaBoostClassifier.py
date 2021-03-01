
from sklearn.ensemble import AdaBoostClassifier as ABClassifier
from MLalgorithms._Classification import Classification


class AdaBoostClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
		self.algorithm = algorithm
		self.random_state = random_state
		self.learning_rate = learning_rate
		self.base_estimator = base_estimator
		self.n_estimators = n_estimators
		self.model = ABClassifier(base_estimator = self.base_estimator,
			random_state = self.random_state,
			algorithm = self.algorithm,
			n_estimators = self.n_estimators,
			learning_rate = self.learning_rate)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

