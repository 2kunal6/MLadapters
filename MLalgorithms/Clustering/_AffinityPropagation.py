
from sklearn.cluster import AffinityPropagation as APClustering
from MLalgorithms._Clustering import Clustering


class AffinityPropagation(Clustering):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False, random_state='warn'):
		self.affinity = affinity
		self.max_iter = max_iter
		self.copy = copy
		self.verbose = verbose
		self.convergence_iter = convergence_iter
		self.random_state = random_state
		self.damping = damping
		self.preference = preference
		self.model = APClustering(copy = self.copy,
			max_iter = self.max_iter,
			verbose = self.verbose,
			random_state = self.random_state,
			damping = self.damping,
			preference = self.preference,
			convergence_iter = self.convergence_iter,
			affinity = self.affinity)

