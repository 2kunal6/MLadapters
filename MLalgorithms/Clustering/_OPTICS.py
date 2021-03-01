
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.min_cluster_size = min_cluster_size
		self.eps = eps
		self.leaf_size = leaf_size
		self.cluster_method = cluster_method
		self.metric_params = metric_params
		self.xi = xi
		self.max_eps = max_eps
		self.p = p
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.min_samples = min_samples
		self.predecessor_correction = predecessor_correction
		self.metric = metric
		self.model = OPTICSClustering(xi = self.xi,
			min_cluster_size = self.min_cluster_size,
			n_jobs = self.n_jobs,
			min_samples = self.min_samples,
			leaf_size = self.leaf_size,
			metric_params = self.metric_params,
			max_eps = self.max_eps,
			p = self.p,
			predecessor_correction = self.predecessor_correction,
			cluster_method = self.cluster_method,
			algorithm = self.algorithm,
			eps = self.eps,
			metric = self.metric)

