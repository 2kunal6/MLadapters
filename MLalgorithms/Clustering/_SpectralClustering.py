
import numpy as np
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):
		self.assign_labels = assign_labels
		self.verbose = verbose
		self.n_clusters = n_clusters
		self.n_init = n_init
		self.degree = degree
		self.affinity = affinity
		self.kernel_params = kernel_params
		self.gamma = gamma
		self.eigen_tol = eigen_tol
		self.n_components = n_components
		self.eigen_solver = eigen_solver
		self.coef0 = coef0
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.n_neighbors = n_neighbors
		self.model = SC(assign_labels = self.assign_labels,
			n_neighbors = self.n_neighbors,
			eigen_solver = self.eigen_solver,
			n_jobs = self.n_jobs,
			kernel_params = self.kernel_params,
			random_state = self.random_state,
			gamma = self.gamma,
			eigen_tol = self.eigen_tol,
			degree = self.degree,
			affinity = self.affinity,
			coef0 = self.coef0,
			verbose = self.verbose,
			n_clusters = self.n_clusters,
			n_components = self.n_components,
			n_init = self.n_init)

