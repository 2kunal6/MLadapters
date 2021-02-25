
from MLalgorithms._MLalgorithms import MLalgorithms


class Regression(MLalgorithms):
	
	def fit(self, X, y, sample_weight=None):
<<<<<<< HEAD
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)
=======
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)
>>>>>>> 0430e16ab26f6fca7bf3d07a8f46b7265698b0cb

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
<<<<<<< HEAD
		self.normalize = normalize
		self.copy_X = copy_X
		self.fit_intercept = fit_intercept
=======
		self.fit_intercept = fit_intercept
		self.copy_X = copy_X
		self.normalize = normalize
>>>>>>> 0430e16ab26f6fca7bf3d07a8f46b7265698b0cb

