
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.leaf_size = leaf_size
		self.p = p
		self.predecessor_correction = predecessor_correction
		self.n_jobs = n_jobs
		self.min_cluster_size = min_cluster_size
		self.max_eps = max_eps
		self.metric = metric
		self.eps = eps
		self.algorithm = algorithm
		self.metric_params = metric_params
		self.min_samples = min_samples
		self.xi = xi
		self.cluster_method = cluster_method
		self.model = OPTICSClustering(metric_params = self.metric_params,
			p = self.p,
			leaf_size = self.leaf_size,
			algorithm = self.algorithm,
			min_samples = self.min_samples,
			n_jobs = self.n_jobs,
			max_eps = self.max_eps,
			eps = self.eps,
			xi = self.xi,
			metric = self.metric,
			min_cluster_size = self.min_cluster_size,
			predecessor_correction = self.predecessor_correction,
			cluster_method = self.cluster_method)

