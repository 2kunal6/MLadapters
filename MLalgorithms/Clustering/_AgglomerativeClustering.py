
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.linkage = linkage
		self.n_clusters = n_clusters
		self.memory = memory
		self.affinity = affinity
		self.compute_distances = compute_distances
		self.connectivity = connectivity
		self.compute_full_tree = compute_full_tree
		self.distance_threshold = distance_threshold
		self.model = AC(memory = self.memory,
			connectivity = self.connectivity,
			n_clusters = self.n_clusters,
			compute_distances = self.compute_distances,
			affinity = self.affinity,
			linkage = self.linkage,
			compute_full_tree = self.compute_full_tree,
			distance_threshold = self.distance_threshold)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

