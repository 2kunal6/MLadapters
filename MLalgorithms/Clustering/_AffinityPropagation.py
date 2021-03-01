
import numpy as np
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.preference = preference
		self.verbose = verbose
		self.affinity = affinity
		self.random_state = random_state
		self.convergence_iter = convergence_iter
		self.damping = damping
		self.copy = copy
		self.max_iter = max_iter
		self.model = APClustering(preference = self.preference,
			copy = self.copy,
			max_iter = self.max_iter,
			convergence_iter = self.convergence_iter,
			damping = self.damping,
			random_state = self.random_state,
			affinity = self.affinity,
			verbose = self.verbose)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

