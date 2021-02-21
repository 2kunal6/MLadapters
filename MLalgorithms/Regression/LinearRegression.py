
from sklearn.linear_model import LinearRegression
from MLalgorithms.Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept, copy_X, n_jobs, normalize):
		Regression.__init__(self, fit_intercept, copy_X, n_jobs, normalize)
		self.model = LinearRegression(n_jobs = self.n_jobs,
			normalize = self.normalize,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept)

