
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.algorithm = algorithm
		self.p = p
		self.metric = metric
		self.weights = weights
		self.n_neighbors = n_neighbors
		self.metric_params = metric_params
		self.n_jobs = n_jobs
		self.leaf_size = leaf_size
		self.model = KNC(metric_params = self.metric_params,
			p = self.p,
			weights = self.weights,
			leaf_size = self.leaf_size,
			n_neighbors = self.n_neighbors,
			metric = self.metric,
			n_jobs = self.n_jobs,
			algorithm = self.algorithm)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

