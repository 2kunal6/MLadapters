
from MLalgorithms._MLalgorithms import MLalgorithms


class Regression(MLalgorithms):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
		self.normalize = normalize
		self.fit_intercept = fit_intercept
		self.copy_X = copy_X

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

