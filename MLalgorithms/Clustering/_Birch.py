
from sklearn.cluster import Birch as BirchClustering
from MLalgorithms._Clustering import Clustering


class Birch(Clustering):
	
	def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
		self.copy = copy
		self.compute_labels = compute_labels
		self.threshold = threshold
		self.branching_factor = branching_factor
		self.n_clusters = n_clusters
		self.model = BirchClustering(branching_factor = self.branching_factor,
			threshold = self.threshold,
			n_clusters = self.n_clusters,
			copy = self.copy,
			compute_labels = self.compute_labels)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

