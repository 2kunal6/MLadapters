
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.memory = memory
		self.affinity = affinity
		self.linkage = linkage
		self.compute_full_tree = compute_full_tree
		self.compute_distances = compute_distances
		self.connectivity = connectivity
		self.distance_threshold = distance_threshold
		self.n_clusters = n_clusters
		self.model = AC(connectivity = self.connectivity,
			compute_full_tree = self.compute_full_tree,
			compute_distances = self.compute_distances,
			memory = self.memory,
			linkage = self.linkage,
			distance_threshold = self.distance_threshold,
			n_clusters = self.n_clusters,
			affinity = self.affinity)

