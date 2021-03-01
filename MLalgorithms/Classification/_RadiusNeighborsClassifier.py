
from sklearn.neighbors import RadiusNeighborsClassifier as RNC
from MLalgorithms._Classification import Classification


class RadiusNeighborsClassifier(Classification):
	
	def __init__(self, radius=1.0, algorithm='auto', leaf_size=30, outlier_label=None, metric_params=None, n_jobs=None):
		self.outlier_label = outlier_label
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.radius = radius
		self.n_jobs = n_jobs
		self.model = RNC(algorithm = self.algorithm,
			leaf_size = self.leaf_size,
			outlier_label = self.outlier_label,
			metric_params = self.metric_params,
			radius = self.radius,
			n_jobs = self.n_jobs)

	def predict(self, X, weights='uniform', p=2, metric='minkowski'):
		return self.model.predict(X=X,
			weights=weights,
			metric=metric,
			p=p)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

