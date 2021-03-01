
from sklearn.neighbors import KNeighborsClassifier as KNC
from MLalgorithms._Classification import Classification


class KNeighborsClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
		self.leaf_size = leaf_size
		self.n_jobs = n_jobs
		self.weights = weights
		self.p = p
		self.metric_params = metric_params
		self.metric = metric
		self.n_neighbors = n_neighbors
		self.algorithm = algorithm
		self.model = KNC(weights = self.weights,
			p = self.p,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			algorithm = self.algorithm,
			metric_params = self.metric_params,
			n_neighbors = self.n_neighbors,
			metric = self.metric)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

