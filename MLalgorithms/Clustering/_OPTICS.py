
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.metric = metric
		self.predecessor_correction = predecessor_correction
		self.p = p
		self.eps = eps
		self.metric_params = metric_params
		self.min_samples = min_samples
		self.n_jobs = n_jobs
		self.cluster_method = cluster_method
		self.min_cluster_size = min_cluster_size
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.xi = xi
		self.max_eps = max_eps
		self.model = OPTICSClustering(predecessor_correction = self.predecessor_correction,
			leaf_size = self.leaf_size,
			cluster_method = self.cluster_method,
			p = self.p,
			max_eps = self.max_eps,
			xi = self.xi,
			metric_params = self.metric_params,
			eps = self.eps,
			metric = self.metric,
			min_samples = self.min_samples,
			min_cluster_size = self.min_cluster_size,
			n_jobs = self.n_jobs,
			algorithm = self.algorithm)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

