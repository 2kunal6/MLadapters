
import numpy as np
from sklearn.mixture import GaussianMixture as GMClustering
from MLalgorithms._Clustering import Clustering


class GaussianMixture(Clustering):
	
	def __init__(self, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
		self.covariance_type = covariance_type
		self.n_init = n_init
		self.random_state = random_state
		self.verbose_interval = verbose_interval
		self.weights_init = weights_init
		self.n_components = n_components
		self.verbose = verbose
		self.reg_covar = reg_covar
		self.init_params = init_params
		self.means_init = means_init
		self.tol = tol
		self.max_iter = max_iter
		self.precisions_init = precisions_init
		self.warm_start = warm_start
		self.model = GMClustering(reg_covar = self.reg_covar,
			weights_init = self.weights_init,
			random_state = self.random_state,
			means_init = self.means_init,
			n_init = self.n_init,
			n_components = self.n_components,
			max_iter = self.max_iter,
			init_params = self.init_params,
			precisions_init = self.precisions_init,
			covariance_type = self.covariance_type,
			verbose = self.verbose,
			warm_start = self.warm_start,
			tol = self.tol,
			verbose_interval = self.verbose_interval)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

