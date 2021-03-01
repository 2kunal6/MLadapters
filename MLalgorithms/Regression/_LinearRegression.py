
from sklearn.linear_model import LinearRegression as LR
from MLalgorithms._Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_x=True, n_jobs=None, positive=False):
		self.positive = positive
		self.n_jobs = n_jobs
		Regression.__init__(self, copy_x=copy_x, fit_intercept=fit_intercept, normalize=normalize)
		self.model = LR(copy_x = self.copy_x,
			n_jobs = self.n_jobs,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			positive = self.positive)

