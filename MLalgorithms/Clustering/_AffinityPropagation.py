
import numpy as np
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.random_state = random_state
		self.convergence_iter = convergence_iter
		self.verbose = verbose
		self.max_iter = max_iter
		self.affinity = affinity
		self.damping = damping
		self.copy = copy
		self.preference = preference
		self.model = APClustering(max_iter = self.max_iter,
			convergence_iter = self.convergence_iter,
			copy = self.copy,
			preference = self.preference,
			random_state = self.random_state,
			affinity = self.affinity,
			damping = self.damping,
			verbose = self.verbose)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

