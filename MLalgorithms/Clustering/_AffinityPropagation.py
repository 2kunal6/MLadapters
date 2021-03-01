
import numpy as np
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.damping = damping
		self.preference = preference
		self.verbose = verbose
		self.copy = copy
		self.affinity = affinity
		self.random_state = random_state
		self.convergence_iter = convergence_iter
		self.max_iter = max_iter
		self.model = APClustering(preference = self.preference,
			random_state = self.random_state,
			max_iter = self.max_iter,
			damping = self.damping,
			copy = self.copy,
			verbose = self.verbose,
			convergence_iter = self.convergence_iter,
			affinity = self.affinity)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

