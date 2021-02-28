
import numpy as np
from sklearn.mixture import GaussianMixture as GMClustering
from MLalgorithms._Clustering import Clustering


class GaussianMixture(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
		self.n_components = n_components
		self.means_init = means_init
		self.random_state = random_state
		self.precisions_init = precisions_init
		self.verbose = verbose
		self.max_iter = max_iter
		self.weights_init = weights_init
		self.verbose_interval = verbose_interval
		self.covariance_type = covariance_type
		self.n_init = n_init
		self.tol = tol
		self.init_params = init_params
		self.warm_start = warm_start
		self.reg_covar = reg_covar
		self.model = GMClustering(weights_init = self.weights_init,
			max_iter = self.max_iter,
			means_init = self.means_init,
			covariance_type = self.covariance_type,
			precisions_init = self.precisions_init,
			n_components = self.n_components,
			random_state = self.random_state,
			n_init = self.n_init,
			warm_start = self.warm_start,
			verbose_interval = self.verbose_interval,
			reg_covar = self.reg_covar,
			tol = self.tol,
			verbose = self.verbose,
			init_params = self.init_params)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

