
import numpy as np
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.preference = preference
		self.convergence_iter = convergence_iter
		self.affinity = affinity
		self.damping = damping
		self.max_iter = max_iter
		self.verbose = verbose
		self.copy = copy
		self.random_state = random_state
		self.model = APClustering(affinity = self.affinity,
			preference = self.preference,
			random_state = self.random_state,
			damping = self.damping,
			max_iter = self.max_iter,
			verbose = self.verbose,
			copy = self.copy,
			convergence_iter = self.convergence_iter)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

