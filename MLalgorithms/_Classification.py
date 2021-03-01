
from MLalgorithms._MLalgorithms import MLalgorithms


class Classification(MLalgorithms):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

