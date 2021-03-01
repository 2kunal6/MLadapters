
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.metric = metric
		self.p = p
		self.metric_params = metric_params
		self.n_neighbors = n_neighbors
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.weights = weights
		self.model = KNC(leaf_size = self.leaf_size,
			p = self.p,
			weights = self.weights,
			metric_params = self.metric_params,
			metric = self.metric,
			algorithm = self.algorithm,
			n_neighbors = self.n_neighbors,
			n_jobs = self.n_jobs)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

