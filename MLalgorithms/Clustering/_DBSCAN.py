
import numpy as np
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.leaf_size = leaf_size
		self.metric = metric
		self.min_samples = min_samples
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.p = p
		self.metric_params = metric_params
		self.eps = eps
		self.model = DBSCANClustering(eps = self.eps,
			leaf_size = self.leaf_size,
			min_samples = self.min_samples,
			p = self.p,
			metric = self.metric,
			n_jobs = self.n_jobs,
			metric_params = self.metric_params,
			algorithm = self.algorithm)

