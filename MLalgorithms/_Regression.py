
from MLalgorithms._MLalgorithms import MLalgorithms


class Regression(MLalgorithms):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

	def __init__(self, fit_intercept=True, normalize=False, copy_x=True):
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.copy_x = copy_x

