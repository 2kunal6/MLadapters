
import numpy as np
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.algorithm = algorithm
		self.eps = eps
		self.p = p
		self.metric_params = metric_params
		self.metric = metric
		self.n_jobs = n_jobs
		self.min_samples = min_samples
		self.leaf_size = leaf_size
		self.model = DBSCANClustering(algorithm = self.algorithm,
			leaf_size = self.leaf_size,
			metric_params = self.metric_params,
			n_jobs = self.n_jobs,
			p = self.p,
			metric = self.metric,
			eps = self.eps,
			min_samples = self.min_samples)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			y=y,
			sample_weight=sample_weight)

