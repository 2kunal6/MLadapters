
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.n_neighbors = n_neighbors
		self.leaf_size = leaf_size
		self.metric = metric
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.weights = weights
		self.metric_params = metric_params
		self.p = p
		self.model = KNC(n_neighbors = self.n_neighbors,
			leaf_size = self.leaf_size,
			weights = self.weights,
			p = self.p,
			metric = self.metric,
			n_jobs = self.n_jobs,
			metric_params = self.metric_params,
			algorithm = self.algorithm)

