
import numpy as np
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):
		self.kernel_params = kernel_params
		self.eigen_solver = eigen_solver
		self.coef0 = coef0
		self.random_state = random_state
		self.verbose = verbose
		self.n_neighbors = n_neighbors
		self.degree = degree
		self.n_jobs = n_jobs
		self.affinity = affinity
		self.n_components = n_components
		self.eigen_tol = eigen_tol
		self.gamma = gamma
		self.n_init = n_init
		self.assign_labels = assign_labels
		self.n_clusters = n_clusters
		self.model = SC(eigen_solver = self.eigen_solver,
			n_clusters = self.n_clusters,
			random_state = self.random_state,
			degree = self.degree,
			n_init = self.n_init,
			n_components = self.n_components,
			gamma = self.gamma,
			verbose = self.verbose,
			eigen_tol = self.eigen_tol,
			kernel_params = self.kernel_params,
			assign_labels = self.assign_labels,
			n_neighbors = self.n_neighbors,
			n_jobs = self.n_jobs,
			affinity = self.affinity,
			coef0 = self.coef0)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

