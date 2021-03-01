
from MLalgorithms._MLalgorithms import MLalgorithms


class Clustering(MLalgorithms):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			y=y,
			sample_weight=sample_weight)

	def predict(self, X, sample_weight=None):
		return self.model.predict(X=X,
			sample_weight=sample_weight)

