
from MLalgorithms._MLalgorithms import MLalgorithms


class Clustering(MLalgorithms):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def predict(self, X, sample_weight=None):
		return self.model.predict(sample_weight=sample_weight,
			X=X)

