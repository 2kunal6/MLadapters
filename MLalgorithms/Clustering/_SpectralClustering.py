
import numpy as np
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):
		self.assign_labels = assign_labels
		self.kernel_params = kernel_params
		self.n_clusters = n_clusters
		self.eigen_tol = eigen_tol
		self.coef0 = coef0
		self.degree = degree
		self.affinity = affinity
		self.gamma = gamma
		self.n_components = n_components
		self.n_jobs = n_jobs
		self.n_neighbors = n_neighbors
		self.random_state = random_state
		self.verbose = verbose
		self.eigen_solver = eigen_solver
		self.n_init = n_init
		self.model = SC(coef0 = self.coef0,
			kernel_params = self.kernel_params,
			affinity = self.affinity,
			random_state = self.random_state,
			n_neighbors = self.n_neighbors,
			n_clusters = self.n_clusters,
			eigen_solver = self.eigen_solver,
			degree = self.degree,
			n_jobs = self.n_jobs,
			gamma = self.gamma,
			verbose = self.verbose,
			assign_labels = self.assign_labels,
			n_init = self.n_init,
			eigen_tol = self.eigen_tol,
			n_components = self.n_components)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

