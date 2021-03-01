
import numpy as np
from sklearn.cluster import KMeans as KMeansClustering
from MLalgorithms._Clustering import Clustering


class KMeans(Clustering):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
		self.max_iter = max_iter
		self.n_clusters = n_clusters
		self.n_init = n_init
		self.verbose = verbose
		self.copy_x = copy_x
		self.tol = tol
		self.random_state = random_state
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.precompute_distances = precompute_distances
		self.model = KMeansClustering(precompute_distances = self.precompute_distances,
			algorithm = self.algorithm,
			random_state = self.random_state,
			copy_x = self.copy_x,
			n_jobs = self.n_jobs,
			verbose = self.verbose,
			n_clusters = self.n_clusters,
			max_iter = self.max_iter,
			tol = self.tol,
			n_init = self.n_init)

	def predict(self, X, sample_weight=None):
		return self.model.predict(sample_weight=sample_weight,
			X=X)

