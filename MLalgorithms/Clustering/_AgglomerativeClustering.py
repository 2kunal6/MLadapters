
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.distance_threshold = distance_threshold
		self.affinity = affinity
		self.compute_full_tree = compute_full_tree
		self.connectivity = connectivity
		self.linkage = linkage
		self.compute_distances = compute_distances
		self.n_clusters = n_clusters
		self.memory = memory
		self.model = AC(n_clusters = self.n_clusters,
			compute_full_tree = self.compute_full_tree,
			distance_threshold = self.distance_threshold,
			linkage = self.linkage,
			compute_distances = self.compute_distances,
			memory = self.memory,
			connectivity = self.connectivity,
			affinity = self.affinity)

