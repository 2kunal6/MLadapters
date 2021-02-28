
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.p = p
		self.algorithm = algorithm
		self.cluster_method = cluster_method
		self.metric = metric
		self.eps = eps
		self.xi = xi
		self.metric_params = metric_params
		self.max_eps = max_eps
		self.min_cluster_size = min_cluster_size
		self.n_jobs = n_jobs
		self.min_samples = min_samples
		self.leaf_size = leaf_size
		self.predecessor_correction = predecessor_correction
		self.model = OPTICSClustering(min_cluster_size = self.min_cluster_size,
			algorithm = self.algorithm,
			cluster_method = self.cluster_method,
			leaf_size = self.leaf_size,
			metric_params = self.metric_params,
			n_jobs = self.n_jobs,
			max_eps = self.max_eps,
			p = self.p,
			metric = self.metric,
			eps = self.eps,
			min_samples = self.min_samples,
			xi = self.xi,
			predecessor_correction = self.predecessor_correction)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

