
from sklearn.neighbors import RadiusNeighborsClassifier as RNC
from MLalgorithms._Classification import Classification


class RadiusNeighborsClassifier(Classification):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, radius=1.0, algorithm='auto', leaf_size=30, outlier_label=None, metric_params=None, n_jobs=None):
		self.radius = radius
		self.leaf_size = leaf_size
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.outlier_label = outlier_label
		self.model = RNC(leaf_size = self.leaf_size,
			metric_params = self.metric_params,
			radius = self.radius,
			n_jobs = self.n_jobs,
			algorithm = self.algorithm,
			outlier_label = self.outlier_label)

	def predict(self, X, weights='uniform', p=2, metric='minkowski'):
		return self.model.predict(p=p,
			weights=weights,
			metric=metric,
			X=X)

