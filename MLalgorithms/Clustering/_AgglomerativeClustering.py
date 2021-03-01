
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.n_clusters = n_clusters
		self.linkage = linkage
		self.connectivity = connectivity
		self.compute_full_tree = compute_full_tree
		self.memory = memory
		self.distance_threshold = distance_threshold
		self.compute_distances = compute_distances
		self.affinity = affinity
		self.model = AC(linkage = self.linkage,
			connectivity = self.connectivity,
			compute_distances = self.compute_distances,
			compute_full_tree = self.compute_full_tree,
			distance_threshold = self.distance_threshold,
			n_clusters = self.n_clusters,
			memory = self.memory,
			affinity = self.affinity)

