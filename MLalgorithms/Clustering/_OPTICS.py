
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.max_eps = max_eps
		self.min_cluster_size = min_cluster_size
		self.leaf_size = leaf_size
		self.xi = xi
		self.metric = metric
		self.min_samples = min_samples
		self.n_jobs = n_jobs
		self.predecessor_correction = predecessor_correction
		self.algorithm = algorithm
		self.cluster_method = cluster_method
		self.metric_params = metric_params
		self.eps = eps
		self.p = p
		self.model = OPTICSClustering(predecessor_correction = self.predecessor_correction,
			eps = self.eps,
			leaf_size = self.leaf_size,
			min_samples = self.min_samples,
			max_eps = self.max_eps,
			min_cluster_size = self.min_cluster_size,
			p = self.p,
			xi = self.xi,
			metric = self.metric,
			n_jobs = self.n_jobs,
			metric_params = self.metric_params,
			cluster_method = self.cluster_method,
			algorithm = self.algorithm)

