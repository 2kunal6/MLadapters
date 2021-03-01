
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.min_cluster_size = min_cluster_size
		self.xi = xi
		self.eps = eps
		self.algorithm = algorithm
		self.cluster_method = cluster_method
		self.min_samples = min_samples
		self.predecessor_correction = predecessor_correction
		self.n_jobs = n_jobs
		self.metric = metric
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.max_eps = max_eps
		self.p = p
		self.model = OPTICSClustering(metric = self.metric,
			predecessor_correction = self.predecessor_correction,
			p = self.p,
			min_samples = self.min_samples,
			n_jobs = self.n_jobs,
			min_cluster_size = self.min_cluster_size,
			eps = self.eps,
			xi = self.xi,
			leaf_size = self.leaf_size,
			max_eps = self.max_eps,
			algorithm = self.algorithm,
			metric_params = self.metric_params,
			cluster_method = self.cluster_method)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

