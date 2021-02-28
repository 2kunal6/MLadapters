
import numpy as np
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):
		self.n_neighbors = n_neighbors
		self.kernel_params = kernel_params
		self.n_components = n_components
		self.gamma = gamma
		self.random_state = random_state
		self.degree = degree
		self.coef0 = coef0
		self.affinity = affinity
		self.eigen_solver = eigen_solver
		self.eigen_tol = eigen_tol
		self.assign_labels = assign_labels
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.n_init = n_init
		self.n_clusters = n_clusters
		self.model = SC(n_neighbors = self.n_neighbors,
			n_clusters = self.n_clusters,
			coef0 = self.coef0,
			gamma = self.gamma,
			affinity = self.affinity,
			n_components = self.n_components,
			n_jobs = self.n_jobs,
			random_state = self.random_state,
			eigen_solver = self.eigen_solver,
			kernel_params = self.kernel_params,
			n_init = self.n_init,
			eigen_tol = self.eigen_tol,
			assign_labels = self.assign_labels,
			verbose = self.verbose,
			degree = self.degree)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

