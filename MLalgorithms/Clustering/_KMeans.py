
from sklearn.cluster import KMeans as KMeansClustering
from MLalgorithms._Clustering import Clustering


class KMeans(Clustering):
	
	def predict(self, X, sample_weight=None):
		return self.model.predict(sample_weight=sample_weight,
			X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
		self.n_init = n_init
		self.verbose = verbose
		self.tol = tol
		self.n_clusters = n_clusters
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.copy_x = copy_x
		self.precompute_distances = precompute_distances
		self.max_iter = max_iter
		self.random_state = random_state
		self.model = KMeansClustering(verbose = self.verbose,
			algorithm = self.algorithm,
			max_iter = self.max_iter,
			n_init = self.n_init,
			tol = self.tol,
			copy_x = self.copy_x,
			random_state = self.random_state,
			n_clusters = self.n_clusters,
			n_jobs = self.n_jobs,
			precompute_distances = self.precompute_distances)

