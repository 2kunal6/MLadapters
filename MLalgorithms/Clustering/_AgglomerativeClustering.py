
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.connectivity = connectivity
		self.linkage = linkage
		self.memory = memory
		self.compute_full_tree = compute_full_tree
		self.distance_threshold = distance_threshold
		self.affinity = affinity
		self.n_clusters = n_clusters
		self.compute_distances = compute_distances
		self.model = AC(linkage = self.linkage,
			n_clusters = self.n_clusters,
			compute_full_tree = self.compute_full_tree,
			memory = self.memory,
			affinity = self.affinity,
			connectivity = self.connectivity,
			distance_threshold = self.distance_threshold,
			compute_distances = self.compute_distances)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

