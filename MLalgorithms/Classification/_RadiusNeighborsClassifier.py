
from sklearn.neighbors import RadiusNeighborsClassifier as RNC
from MLalgorithms._Classification import Classification


class RadiusNeighborsClassifier(Classification):
	
	def predict(self, X, weights='uniform', p=2, metric='minkowski'):
		return self.model.predict(X=X,
			weights=weights,
			metric=metric,
			p=p)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def __init__(self, radius=1.0, algorithm='auto', leaf_size=30, outlier_label=None, metric_params=None, n_jobs=None):
		self.leaf_size = leaf_size
		self.outlier_label = outlier_label
		self.n_jobs = n_jobs
		self.algorithm = algorithm
		self.metric_params = metric_params
		self.radius = radius
		self.model = RNC(leaf_size = self.leaf_size,
			outlier_label = self.outlier_label,
			n_jobs = self.n_jobs,
			radius = self.radius,
			metric_params = self.metric_params,
			algorithm = self.algorithm)

