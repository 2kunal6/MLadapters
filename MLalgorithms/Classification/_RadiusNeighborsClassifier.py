
from sklearn.neighbors import RadiusNeighborsClassifier as RNC
from MLalgorithms._Classification import Classification


class RadiusNeighborsClassifier(Classification):
	
	def __init__(self, radius=1.0, algorithm='auto', leaf_size=30, outlier_label=None, metric_params=None, n_jobs=None):
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.radius = radius
		self.outlier_label = outlier_label
		self.model = RNC(algorithm = self.algorithm,
			outlier_label = self.outlier_label,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			radius = self.radius,
			metric_params = self.metric_params)

	def predict(self, X, weights='uniform', p=2, metric='minkowski'):
		return self.model.predict(metric=metric,
			weights=weights,
			p=p,
			X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

