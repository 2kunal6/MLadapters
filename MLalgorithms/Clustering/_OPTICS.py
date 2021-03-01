
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.max_eps = max_eps
		self.metric = metric
		self.eps = eps
		self.min_samples = min_samples
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.predecessor_correction = predecessor_correction
		self.min_cluster_size = min_cluster_size
		self.cluster_method = cluster_method
		self.xi = xi
		self.p = p
		self.model = OPTICSClustering(eps = self.eps,
			max_eps = self.max_eps,
			min_samples = self.min_samples,
			predecessor_correction = self.predecessor_correction,
			algorithm = self.algorithm,
			p = self.p,
			cluster_method = self.cluster_method,
			min_cluster_size = self.min_cluster_size,
			xi = self.xi,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			metric = self.metric,
			metric_params = self.metric_params)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

