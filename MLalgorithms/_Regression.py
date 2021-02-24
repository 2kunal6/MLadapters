
from MLalgorithms._MLalgorithms import MLalgorithms


class Regression(MLalgorithms):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
		self.copy_X = copy_X
		self.normalize = normalize
		self.fit_intercept = fit_intercept

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			X=X,
			sample_weight=sample_weight)

