
from sklearn.neighbors import RadiusNeighborsClassifier as RNC
from MLalgorithms._Classification import Classification


class RadiusNeighborsClassifier(Classification):
	
	def predict(self, X, weights='uniform', p=2, metric='minkowski'):
		return self.model.predict(weights=weights,
			X=X,
			p=p,
			metric=metric)

	def __init__(self, radius=1.0, algorithm='auto', leaf_size=30, outlier_label=None, metric_params=None, n_jobs=None):
		self.algorithm = algorithm
		self.outlier_label = outlier_label
		self.n_jobs = n_jobs
		self.radius = radius
		self.metric_params = metric_params
		self.leaf_size = leaf_size
		self.model = RNC(n_jobs = self.n_jobs,
			leaf_size = self.leaf_size,
			outlier_label = self.outlier_label,
			algorithm = self.algorithm,
			metric_params = self.metric_params,
			radius = self.radius)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

