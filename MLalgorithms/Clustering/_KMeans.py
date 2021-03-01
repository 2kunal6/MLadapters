
import numpy as np
from sklearn.cluster import KMeans as KMeansClustering
from MLalgorithms._Clustering import Clustering


class KMeans(Clustering):
	
	def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
		self.n_clusters = n_clusters
		self.precompute_distances = precompute_distances
		self.random_state = random_state
		self.verbose = verbose
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.tol = tol
		self.copy_x = copy_x
		self.n_init = n_init
		self.max_iter = max_iter
		self.model = KMeansClustering(max_iter = self.max_iter,
			n_clusters = self.n_clusters,
			algorithm = self.algorithm,
			n_init = self.n_init,
			precompute_distances = self.precompute_distances,
			tol = self.tol,
			random_state = self.random_state,
			n_jobs = self.n_jobs,
			copy_x = self.copy_x,
			verbose = self.verbose)

	def predict(self, X, sample_weight=None):
		return self.model.predict(sample_weight=sample_weight,
			X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

