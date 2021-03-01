
from sklearn.cluster import KMeans as KMeansClustering
from MLalgorithms._Clustering import Clustering


class KMeans(Clustering):
	
	def predict(self, X, sample_weight=None):
		return self.model.predict(X=X,
			sample_weight=sample_weight)

	def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
		self.max_iter = max_iter
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.n_clusters = n_clusters
		self.verbose = verbose
		self.tol = tol
		self.copy_x = copy_x
		self.algorithm = algorithm
		self.n_init = n_init
		self.precompute_distances = precompute_distances
		self.model = KMeansClustering(tol = self.tol,
			precompute_distances = self.precompute_distances,
			n_jobs = self.n_jobs,
			random_state = self.random_state,
			copy_x = self.copy_x,
			algorithm = self.algorithm,
			max_iter = self.max_iter,
			verbose = self.verbose,
			n_init = self.n_init,
			n_clusters = self.n_clusters)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

