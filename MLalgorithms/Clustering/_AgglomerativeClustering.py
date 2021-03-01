
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.connectivity = connectivity
		self.memory = memory
		self.compute_full_tree = compute_full_tree
		self.distance_threshold = distance_threshold
		self.affinity = affinity
		self.n_clusters = n_clusters
		self.compute_distances = compute_distances
		self.linkage = linkage
		self.model = AC(linkage = self.linkage,
			affinity = self.affinity,
			connectivity = self.connectivity,
			compute_full_tree = self.compute_full_tree,
			compute_distances = self.compute_distances,
			memory = self.memory,
			n_clusters = self.n_clusters,
			distance_threshold = self.distance_threshold)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

