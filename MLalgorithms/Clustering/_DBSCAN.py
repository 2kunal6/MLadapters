
import numpy as np
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.metric = metric
		self.algorithm = algorithm
		self.p = p
		self.eps = eps
		self.min_samples = min_samples
		self.metric_params = metric_params
		self.n_jobs = n_jobs
		self.leaf_size = leaf_size
		self.model = DBSCANClustering(n_jobs = self.n_jobs,
			metric_params = self.metric_params,
			p = self.p,
			eps = self.eps,
			leaf_size = self.leaf_size,
			metric = self.metric,
			min_samples = self.min_samples,
			algorithm = self.algorithm)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			y=y,
			sample_weight=sample_weight)

