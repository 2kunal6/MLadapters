
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.affinity = affinity
		self.copy = copy
		self.random_state = random_state
		self.verbose = verbose
		self.damping = damping
		self.preference = preference
		self.convergence_iter = convergence_iter
		self.max_iter = max_iter
		self.model = APClustering(convergence_iter = self.convergence_iter,
			preference = self.preference,
			verbose = self.verbose,
			affinity = self.affinity,
			damping = self.damping,
			max_iter = self.max_iter,
			random_state = self.random_state,
			copy = self.copy)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

