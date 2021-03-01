
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.algorithm = algorithm
		self.p = p
		self.cluster_method = cluster_method
		self.metric = metric
		self.eps = eps
		self.max_eps = max_eps
		self.xi = xi
		self.min_samples = min_samples
		self.predecessor_correction = predecessor_correction
		self.metric_params = metric_params
		self.min_cluster_size = min_cluster_size
		self.n_jobs = n_jobs
		self.leaf_size = leaf_size
		self.model = OPTICSClustering(n_jobs = self.n_jobs,
			cluster_method = self.cluster_method,
			metric_params = self.metric_params,
			xi = self.xi,
			max_eps = self.max_eps,
			min_cluster_size = self.min_cluster_size,
			p = self.p,
			eps = self.eps,
			leaf_size = self.leaf_size,
			predecessor_correction = self.predecessor_correction,
			metric = self.metric,
			min_samples = self.min_samples,
			algorithm = self.algorithm)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

