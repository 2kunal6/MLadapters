
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.n_neighbors = n_neighbors
		self.metric_params = metric_params
		self.weights = weights
		self.algorithm = algorithm
		self.p = p
		self.leaf_size = leaf_size
		self.metric = metric
		self.n_jobs = n_jobs
		self.model = KNC(algorithm = self.algorithm,
			leaf_size = self.leaf_size,
			metric_params = self.metric_params,
			metric = self.metric,
			weights = self.weights,
			n_neighbors = self.n_neighbors,
			p = self.p,
			n_jobs = self.n_jobs)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

