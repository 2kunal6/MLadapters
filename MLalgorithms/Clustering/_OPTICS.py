
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.eps = eps
		self.xi = xi
		self.min_cluster_size = min_cluster_size
		self.algorithm = algorithm
		self.predecessor_correction = predecessor_correction
		self.metric = metric
		self.p = p
		self.min_samples = min_samples
		self.max_eps = max_eps
		self.n_jobs = n_jobs
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.cluster_method = cluster_method
		self.model = OPTICSClustering(p = self.p,
			xi = self.xi,
			metric_params = self.metric_params,
			min_cluster_size = self.min_cluster_size,
			max_eps = self.max_eps,
			algorithm = self.algorithm,
			metric = self.metric,
			leaf_size = self.leaf_size,
			min_samples = self.min_samples,
			n_jobs = self.n_jobs,
			eps = self.eps,
			predecessor_correction = self.predecessor_correction,
			cluster_method = self.cluster_method)

