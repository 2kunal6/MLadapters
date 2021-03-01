
from sklearn.cluster import DBSCAN as DBSCANClustering
from MLalgorithms._Clustering import Clustering


class DBSCAN(Clustering):
	
	def __init__(self, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
		self.metric_params = metric_params
		self.min_samples = min_samples
		self.eps = eps
		self.n_jobs = n_jobs
		self.metric = metric
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.model = DBSCANClustering(leaf_size = self.leaf_size,
			p = self.p,
			eps = self.eps,
			metric_params = self.metric_params,
			metric = self.metric,
			min_samples = self.min_samples,
			algorithm = self.algorithm,
			n_jobs = self.n_jobs)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

