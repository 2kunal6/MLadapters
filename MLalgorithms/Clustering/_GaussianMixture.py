
import numpy as np
from sklearn.mixture import GaussianMixture as GMClustering
from MLalgorithms._Clustering import Clustering


class GaussianMixture(Clustering):
	
	def __init__(self, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
		self.means_init = means_init
		self.init_params = init_params
		self.n_init = n_init
		self.tol = tol
		self.precisions_init = precisions_init
		self.max_iter = max_iter
		self.verbose_interval = verbose_interval
		self.random_state = random_state
		self.reg_covar = reg_covar
		self.covariance_type = covariance_type
		self.verbose = verbose
		self.warm_start = warm_start
		self.weights_init = weights_init
		self.n_components = n_components
		self.model = GMClustering(max_iter = self.max_iter,
			warm_start = self.warm_start,
			precisions_init = self.precisions_init,
			weights_init = self.weights_init,
			n_init = self.n_init,
			means_init = self.means_init,
			verbose_interval = self.verbose_interval,
			init_params = self.init_params,
			tol = self.tol,
			random_state = self.random_state,
			n_components = self.n_components,
			verbose = self.verbose,
			covariance_type = self.covariance_type,
			reg_covar = self.reg_covar)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

