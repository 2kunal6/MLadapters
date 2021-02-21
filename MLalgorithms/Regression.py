
import MLalgorithms


class Regression(MLalgorithms):
	
	def fit(self, y, sample_weight, X):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, fit_intercept, copy_X, n_jobs = None, normalize):
		self.fit_intercept = fit_intercept
		self.copy_X = copy_X
		self.n_jobs = n_jobs
		self.normalize = normalize

