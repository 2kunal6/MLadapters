
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.affinity = affinity
		self.distance_threshold = distance_threshold
		self.linkage = linkage
		self.n_clusters = n_clusters
		self.connectivity = connectivity
		self.memory = memory
		self.compute_distances = compute_distances
		self.compute_full_tree = compute_full_tree
		self.model = AC(compute_distances = self.compute_distances,
			compute_full_tree = self.compute_full_tree,
			connectivity = self.connectivity,
			distance_threshold = self.distance_threshold,
			linkage = self.linkage,
			affinity = self.affinity,
			n_clusters = self.n_clusters,
			memory = self.memory)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

