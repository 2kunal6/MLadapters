
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier as ABClassifier
from MLalgorithms._Classification import Classification


class AdaBoostClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.algorithm = algorithm
		self.base_estimator = base_estimator
		self.random_state = random_state
		self.model = ABClassifier(random_state = self.random_state,
			learning_rate = self.learning_rate,
			algorithm = self.algorithm,
			n_estimators = self.n_estimators,
			base_estimator = self.base_estimator)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

