
from sklearn.cluster import Birch as BirchClustering
from MLalgorithms._Clustering import Clustering


class Birch(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
		self.branching_factor = branching_factor
		self.n_clusters = n_clusters
		self.threshold = threshold
		self.copy = copy
		self.compute_labels = compute_labels
		self.model = BirchClustering(compute_labels = self.compute_labels,
			copy = self.copy,
			threshold = self.threshold,
			n_clusters = self.n_clusters,
			branching_factor = self.branching_factor)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

