
from MLalgorithms._MLalgorithms import MLalgorithms


class Classification(MLalgorithms):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def predict(self, X):
		return self.model.predict(X=X)

