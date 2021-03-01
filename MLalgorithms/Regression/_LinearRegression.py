
from sklearn.linear_model import LinearRegression as LR
from MLalgorithms._Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, positive=False):
		self.copy_X = copy_X
		self.n_jobs = n_jobs
		self.positive = positive
		self.normalize = normalize
		self.fit_intercept = fit_intercept
		self.model = LR(copy_X = self.copy_X,
			n_jobs = self.n_jobs,
			normalize = self.normalize,
			positive = self.positive,
			fit_intercept = self.fit_intercept)

