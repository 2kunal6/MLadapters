
from sklearn.neighbors import NearestCentroid as NCC
from MLalgorithms._Classification import Classification


class NearestCentroid(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, metric='euclidean', shrink_threshold=None):
		self.shrink_threshold = shrink_threshold
		self.metric = metric
		self.model = NCC(shrink_threshold = self.shrink_threshold,
			metric = self.metric)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

