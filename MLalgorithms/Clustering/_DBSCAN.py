
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.eps = eps
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.p = p
		self.metric_params = metric_params
		self.metric = metric
		self.min_samples = min_samples
		self.leaf_size = leaf_size
		self.model = DBSCANClustering(p = self.p,
			leaf_size = self.leaf_size,
			algorithm = self.algorithm,
			min_samples = self.min_samples,
			metric_params = self.metric_params,
			eps = self.eps,
			n_jobs = self.n_jobs,
			metric = self.metric)

