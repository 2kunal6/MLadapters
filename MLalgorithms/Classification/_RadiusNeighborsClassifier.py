
from sklearn.neighbors import RadiusNeighborsClassifier as RNC
from MLalgorithms._Classification import Classification


class RadiusNeighborsClassifier(Classification):
	
	def __init__(self, radius=1.0, algorithm='auto', leaf_size=30, outlier_label=None, metric_params=None, n_jobs=None):
		self.algorithm = algorithm
		self.outlier_label = outlier_label
		self.metric_params = metric_params
		self.n_jobs = n_jobs
		self.radius = radius
		self.leaf_size = leaf_size
		self.model = RNC(metric_params = self.metric_params,
			outlier_label = self.outlier_label,
			radius = self.radius,
			leaf_size = self.leaf_size,
			n_jobs = self.n_jobs,
			algorithm = self.algorithm)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X, weights='uniform', p=2, metric='minkowski'):
		return self.model.predict(metric=metric,
			X=X,
			p=p,
			weights=weights)

