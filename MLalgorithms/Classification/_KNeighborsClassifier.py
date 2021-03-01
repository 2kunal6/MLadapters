
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.algorithm = algorithm
		self.weights = weights
		self.metric = metric
		self.p = p
		self.n_jobs = n_jobs
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.n_neighbors = n_neighbors
		self.model = KNC(metric_params = self.metric_params,
			metric = self.metric,
			algorithm = self.algorithm,
			n_neighbors = self.n_neighbors,
			weights = self.weights,
			n_jobs = self.n_jobs,
			leaf_size = self.leaf_size,
			p = self.p)

