
from sklearn.cluster import Birch as BirchClustering
from MLalgorithms._Clustering import Clustering


class Birch(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
		self.threshold = threshold
		self.copy = copy
		self.n_clusters = n_clusters
		self.compute_labels = compute_labels
		self.branching_factor = branching_factor
		self.model = BirchClustering(n_clusters = self.n_clusters,
			threshold = self.threshold,
			compute_labels = self.compute_labels,
			branching_factor = self.branching_factor,
			copy = self.copy)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

