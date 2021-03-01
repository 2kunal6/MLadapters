
import numpy as np
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.random_state = random_state
		self.verbose = verbose
		self.convergence_iter = convergence_iter
		self.affinity = affinity
		self.preference = preference
		self.damping = damping
		self.max_iter = max_iter
		self.copy = copy
		self.model = APClustering(preference = self.preference,
			damping = self.damping,
			random_state = self.random_state,
			affinity = self.affinity,
			copy = self.copy,
			verbose = self.verbose,
			convergence_iter = self.convergence_iter,
			max_iter = self.max_iter)

	def predict(self, X):
		return self.model.predict(X=X)

