
import numpy as np
from sklearn.cluster import KMeans as KMeansClustering
from MLalgorithms._Clustering import Clustering


class KMeans(Clustering):
	
	def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
		self.max_iter = max_iter
		self.random_state = random_state
		self.verbose = verbose
		self.n_jobs = n_jobs
		self.tol = tol
		self.algorithm = algorithm
		self.precompute_distances = precompute_distances
		self.n_init = n_init
		self.copy_x = copy_x
		self.n_clusters = n_clusters
		self.model = KMeansClustering(algorithm = self.algorithm,
			random_state = self.random_state,
			n_init = self.n_init,
			precompute_distances = self.precompute_distances,
			max_iter = self.max_iter,
			copy_x = self.copy_x,
			verbose = self.verbose,
			tol = self.tol,
			n_jobs = self.n_jobs,
			n_clusters = self.n_clusters)

	def predict(self, X, sample_weight=None):
		return self.model.predict(sample_weight=sample_weight,
			X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

