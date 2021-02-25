
from MLalgorithms._MLalgorithms import MLalgorithms


class Regression(MLalgorithms):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
		self.copy_X = copy_X
		self.normalize = normalize
		self.fit_intercept = fit_intercept

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def predict(self, X):
		return self.model.predict(X=X)

