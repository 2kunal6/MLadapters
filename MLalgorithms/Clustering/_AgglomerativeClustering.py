
import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC
from MLalgorithms._Clustering import Clustering


class AgglomerativeClustering(Clustering):
	
	def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
		self.linkage = linkage
		self.memory = memory
		self.affinity = affinity
		self.compute_distances = compute_distances
		self.compute_full_tree = compute_full_tree
		self.connectivity = connectivity
		self.distance_threshold = distance_threshold
		self.n_clusters = n_clusters
		self.model = AC(compute_distances = self.compute_distances,
			distance_threshold = self.distance_threshold,
			n_clusters = self.n_clusters,
			linkage = self.linkage,
			affinity = self.affinity,
			connectivity = self.connectivity,
			memory = self.memory,
			compute_full_tree = self.compute_full_tree)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

