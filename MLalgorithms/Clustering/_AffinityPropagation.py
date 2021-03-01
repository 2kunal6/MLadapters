
import numpy as np
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.affinity = affinity
		self.verbose = verbose
		self.preference = preference
		self.copy = copy
		self.max_iter = max_iter
		self.convergence_iter = convergence_iter
		self.random_state = random_state
		self.damping = damping
		self.model = APClustering(max_iter = self.max_iter,
			verbose = self.verbose,
			convergence_iter = self.convergence_iter,
			copy = self.copy,
			damping = self.damping,
			affinity = self.affinity,
			random_state = self.random_state,
			preference = self.preference)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

