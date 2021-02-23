
from sklearn.linear_model import LinearRegression as LR
from MLalgorithms._Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept = True, normalize = False, copy_X = True, n_jobs = None, positive = False):
		self.n_jobs = n_jobs
		self.positive = positive
		Regression.__init__(self, fit_intercept, normalize, copy_X)
		self.model = LR(fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			normalize = self.normalize,
			n_jobs = self.n_jobs,
			positive = self.positive)

