
from sklearn.linear_model import LinearRegression as LR
from MLalgorithms.Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept, normalize, copy_X, positive = False, n_jobs = None):
		self.positive = positive
		self.n_jobs = n_jobs
		Regression.__init__(self, fit_intercept, normalize, copy_X)
		self.model = LR(n_jobs = self.n_jobs,
			positive = self.positive,
			copy_X = self.copy_X,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept)

