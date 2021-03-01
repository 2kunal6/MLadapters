
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.predecessor_correction = predecessor_correction
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.max_eps = max_eps
		self.p = p
		self.metric_params = metric_params
		self.eps = eps
		self.metric = metric
		self.min_samples = min_samples
		self.min_cluster_size = min_cluster_size
		self.leaf_size = leaf_size
		self.cluster_method = cluster_method
		self.xi = xi
		self.model = OPTICSClustering(max_eps = self.max_eps,
			p = self.p,
			leaf_size = self.leaf_size,
			algorithm = self.algorithm,
			min_samples = self.min_samples,
			predecessor_correction = self.predecessor_correction,
			metric_params = self.metric_params,
			cluster_method = self.cluster_method,
			min_cluster_size = self.min_cluster_size,
			xi = self.xi,
			eps = self.eps,
			n_jobs = self.n_jobs,
			metric = self.metric)

