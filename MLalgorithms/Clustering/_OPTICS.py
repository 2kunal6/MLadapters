
import numpy as np
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.cluster_method = cluster_method
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.algorithm = algorithm
		self.p = p
		self.metric = metric
		self.min_samples = min_samples
		self.min_cluster_size = min_cluster_size
		self.n_jobs = n_jobs
		self.xi = xi
		self.eps = eps
		self.predecessor_correction = predecessor_correction
		self.max_eps = max_eps
		self.model = OPTICSClustering(metric_params = self.metric_params,
			min_samples = self.min_samples,
			min_cluster_size = self.min_cluster_size,
			algorithm = self.algorithm,
			predecessor_correction = self.predecessor_correction,
			xi = self.xi,
			n_jobs = self.n_jobs,
			cluster_method = self.cluster_method,
			max_eps = self.max_eps,
			leaf_size = self.leaf_size,
			p = self.p,
			eps = self.eps,
			metric = self.metric)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

