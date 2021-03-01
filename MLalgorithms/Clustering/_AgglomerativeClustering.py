
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.n_clusters = n_clusters
		self.connectivity = connectivity
		self.distance_threshold = distance_threshold
		self.linkage = linkage
		self.compute_full_tree = compute_full_tree
		self.affinity = affinity
		self.compute_distances = compute_distances
		self.memory = memory
		self.model = AC(memory = self.memory,
			affinity = self.affinity,
			connectivity = self.connectivity,
			n_clusters = self.n_clusters,
			linkage = self.linkage,
			compute_full_tree = self.compute_full_tree,
			distance_threshold = self.distance_threshold,
			compute_distances = self.compute_distances)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

