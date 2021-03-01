
from sklearn.cluster import SpectralClustering as SC
from MLalgorithms._Clustering import Clustering


class SpectralClustering(Clustering):
	
	def __init__(self, n_clusters=8, eigen_solver='None', n_components='n_clusters', random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False):
		self.n_init = n_init
		self.degree = degree
		self.eigen_tol = eigen_tol
		self.n_clusters = n_clusters
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.random_state = random_state
		self.assign_labels = assign_labels
		self.coef0 = coef0
		self.kernel_params = kernel_params
		self.n_neighbors = n_neighbors
		self.n_components = n_components
		self.affinity = affinity
		self.eigen_solver = eigen_solver
		self.gamma = gamma
		self.model = SC(eigen_solver = self.eigen_solver,
			n_clusters = self.n_clusters,
			n_init = self.n_init,
			verbose = self.verbose,
			affinity = self.affinity,
			kernel_params = self.kernel_params,
			gamma = self.gamma,
			random_state = self.random_state,
			coef0 = self.coef0,
			n_components = self.n_components,
			assign_labels = self.assign_labels,
			n_neighbors = self.n_neighbors,
			eigen_tol = self.eigen_tol,
			n_jobs = self.n_jobs,
			degree = self.degree)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

