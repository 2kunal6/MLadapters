
import numpy as np
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.p = p
		self.algorithm = algorithm
		self.min_samples = min_samples
		self.n_jobs = n_jobs
		self.eps = eps
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.metric = metric
		self.model = DBSCANClustering(metric = self.metric,
			min_samples = self.min_samples,
			n_jobs = self.n_jobs,
			eps = self.eps,
			leaf_size = self.leaf_size,
			algorithm = self.algorithm,
			metric_params = self.metric_params,
			p = self.p)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

