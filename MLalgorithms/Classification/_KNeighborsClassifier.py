
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.metric = metric
		self.metric_params = metric_params
		self.n_neighbors = n_neighbors
		self.leaf_size = leaf_size
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.weights = weights
		self.p = p
		self.model = KNC(n_neighbors = self.n_neighbors,
			p = self.p,
			algorithm = self.algorithm,
			weights = self.weights,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			metric = self.metric,
			metric_params = self.metric_params)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

