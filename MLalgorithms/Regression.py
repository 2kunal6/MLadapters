
import MLalgorithms


class Regression(MLalgorithms):
	
	def __init__(self, fit_intercept = True, normalize = False, copy_X = True):
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.copy_X = copy_X

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight = None):
		return self.model.fit(X=X,
			y=y,
			sample_weight=sample_weight)

