
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.distance_threshold = distance_threshold
		self.affinity = affinity
		self.compute_full_tree = compute_full_tree
		self.memory = memory
		self.compute_distances = compute_distances
		self.linkage = linkage
		self.connectivity = connectivity
		self.n_clusters = n_clusters
		self.model = AC(compute_distances = self.compute_distances,
			linkage = self.linkage,
			distance_threshold = self.distance_threshold,
			compute_full_tree = self.compute_full_tree,
			connectivity = self.connectivity,
			memory = self.memory,
			affinity = self.affinity,
			n_clusters = self.n_clusters)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

