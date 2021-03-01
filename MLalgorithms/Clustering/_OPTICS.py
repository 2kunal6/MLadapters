
from sklearn.cluster import OPTICS as OPTICSClustering
from MLalgorithms._Clustering import Clustering


class OPTICS(Clustering):
	
	def __init__(self, min_samples=5, max_eps='inf', metric='minkowski', p=2, metric_params=None, cluster_method='xi', eps=None, xi=0.05, predecessor_correction=True, min_cluster_size=None, algorithm='auto', leaf_size=30, n_jobs=None):
		self.leaf_size = leaf_size
		self.eps = eps
		self.max_eps = max_eps
		self.min_cluster_size = min_cluster_size
		self.n_jobs = n_jobs
		self.min_samples = min_samples
		self.cluster_method = cluster_method
		self.xi = xi
		self.p = p
		self.metric_params = metric_params
		self.metric = metric
		self.algorithm = algorithm
		self.predecessor_correction = predecessor_correction
		self.model = OPTICSClustering(eps = self.eps,
			min_cluster_size = self.min_cluster_size,
			p = self.p,
			xi = self.xi,
			leaf_size = self.leaf_size,
			max_eps = self.max_eps,
			n_jobs = self.n_jobs,
			algorithm = self.algorithm,
			predecessor_correction = self.predecessor_correction,
			metric_params = self.metric_params,
			min_samples = self.min_samples,
			cluster_method = self.cluster_method,
			metric = self.metric)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

