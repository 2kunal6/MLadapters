
from MLalgorithms._MLalgorithms import MLalgorithms


class Regression(MLalgorithms):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_x=True):
		self.normalize = normalize
		self.copy_x = copy_x
		self.fit_intercept = fit_intercept

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			y=y,
			sample_weight=sample_weight)

	def predict(self, X):
		return self.model.predict(X=X)

