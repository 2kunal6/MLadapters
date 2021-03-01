
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.leaf_size = leaf_size
		self.eps = eps
		self.n_jobs = n_jobs
		self.metric = metric
		self.p = p
		self.algorithm = algorithm
		self.metric_params = metric_params
		self.min_samples = min_samples
		self.model = DBSCANClustering(metric_params = self.metric_params,
			p = self.p,
			leaf_size = self.leaf_size,
			algorithm = self.algorithm,
			min_samples = self.min_samples,
			n_jobs = self.n_jobs,
			eps = self.eps,
			metric = self.metric)

