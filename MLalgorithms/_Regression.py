
from MLalgorithms._MLalgorithms import MLalgorithms


class Regression(MLalgorithms):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
		self.fit_intercept = fit_intercept
		self.copy_X = copy_X
		self.normalize = normalize

