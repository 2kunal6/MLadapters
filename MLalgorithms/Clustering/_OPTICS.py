
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.max_eps = max_eps
		self.eps = eps
		self.min_samples = min_samples
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.predecessor_correction = predecessor_correction
		self.p = p
		self.xi = xi
		self.leaf_size = leaf_size
		self.cluster_method = cluster_method
		self.metric = metric
		self.min_cluster_size = min_cluster_size
		self.n_jobs = n_jobs
		self.model = OPTICSClustering(algorithm = self.algorithm,
			leaf_size = self.leaf_size,
			eps = self.eps,
			max_eps = self.max_eps,
			xi = self.xi,
			cluster_method = self.cluster_method,
			metric_params = self.metric_params,
			min_cluster_size = self.min_cluster_size,
			metric = self.metric,
			predecessor_correction = self.predecessor_correction,
			p = self.p,
			n_jobs = self.n_jobs,
			min_samples = self.min_samples)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

