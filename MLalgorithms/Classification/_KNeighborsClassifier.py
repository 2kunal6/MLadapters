
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.leaf_size = leaf_size
		self.p = p
		self.n_neighbors = n_neighbors
		self.n_jobs = n_jobs
		self.weights = weights
		self.metric = metric
		self.algorithm = algorithm
		self.metric_params = metric_params
		self.model = KNC(metric_params = self.metric_params,
			p = self.p,
			leaf_size = self.leaf_size,
			algorithm = self.algorithm,
			n_jobs = self.n_jobs,
			weights = self.weights,
			metric = self.metric,
			n_neighbors = self.n_neighbors)

