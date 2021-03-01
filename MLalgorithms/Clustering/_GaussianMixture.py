
import numpy as np
from sklearn.mixture import GaussianMixture as GMClustering
from MLalgorithms._Clustering import Clustering


class GaussianMixture(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
		self.n_components = n_components
		self.means_init = means_init
		self.verbose = verbose
		self.warm_start = warm_start
		self.weights_init = weights_init
		self.init_params = init_params
		self.max_iter = max_iter
		self.covariance_type = covariance_type
		self.precisions_init = precisions_init
		self.random_state = random_state
		self.verbose_interval = verbose_interval
		self.tol = tol
		self.n_init = n_init
		self.reg_covar = reg_covar
		self.model = GMClustering(reg_covar = self.reg_covar,
			init_params = self.init_params,
			max_iter = self.max_iter,
			verbose_interval = self.verbose_interval,
			means_init = self.means_init,
			warm_start = self.warm_start,
			random_state = self.random_state,
			weights_init = self.weights_init,
			verbose = self.verbose,
			n_components = self.n_components,
			n_init = self.n_init,
			precisions_init = self.precisions_init,
			covariance_type = self.covariance_type,
			tol = self.tol)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

