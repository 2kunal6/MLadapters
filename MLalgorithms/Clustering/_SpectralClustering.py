
import numpy as np
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):
		self.degree = degree
		self.eigen_tol = eigen_tol
		self.coef0 = coef0
		self.verbose = verbose
		self.kernel_params = kernel_params
		self.n_clusters = n_clusters
		self.n_components = n_components
		self.n_neighbors = n_neighbors
		self.assign_labels = assign_labels
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.affinity = affinity
		self.n_init = n_init
		self.gamma = gamma
		self.eigen_solver = eigen_solver
		self.model = SC(n_neighbors = self.n_neighbors,
			degree = self.degree,
			n_clusters = self.n_clusters,
			n_jobs = self.n_jobs,
			random_state = self.random_state,
			affinity = self.affinity,
			coef0 = self.coef0,
			kernel_params = self.kernel_params,
			verbose = self.verbose,
			n_components = self.n_components,
			gamma = self.gamma,
			n_init = self.n_init,
			eigen_solver = self.eigen_solver,
			eigen_tol = self.eigen_tol,
			assign_labels = self.assign_labels)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

