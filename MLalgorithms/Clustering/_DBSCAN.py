
import numpy as np
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.eps = eps
		self.p = p
		self.min_samples = min_samples
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.metric = metric
		self.n_jobs = n_jobs
		self.model = DBSCANClustering(algorithm = self.algorithm,
			leaf_size = self.leaf_size,
			eps = self.eps,
			metric_params = self.metric_params,
			metric = self.metric,
			p = self.p,
			n_jobs = self.n_jobs,
			min_samples = self.min_samples)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

