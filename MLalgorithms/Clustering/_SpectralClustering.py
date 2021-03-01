
import numpy as np
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):
		self.verbose = verbose
		self.random_state = random_state
		self.eigen_tol = eigen_tol
		self.assign_labels = assign_labels
		self.gamma = gamma
		self.n_clusters = n_clusters
		self.n_neighbors = n_neighbors
		self.n_components = n_components
		self.kernel_params = kernel_params
		self.degree = degree
		self.coef0 = coef0
		self.n_init = n_init
		self.n_jobs = n_jobs
		self.affinity = affinity
		self.eigen_solver = eigen_solver
		self.model = SC(n_jobs = self.n_jobs,
			degree = self.degree,
			verbose = self.verbose,
			n_clusters = self.n_clusters,
			coef0 = self.coef0,
			eigen_solver = self.eigen_solver,
			n_components = self.n_components,
			gamma = self.gamma,
			n_init = self.n_init,
			affinity = self.affinity,
			kernel_params = self.kernel_params,
			random_state = self.random_state,
			assign_labels = self.assign_labels,
			n_neighbors = self.n_neighbors,
			eigen_tol = self.eigen_tol)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

