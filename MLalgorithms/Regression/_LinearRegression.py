
from sklearn.linear_model import LinearRegression as LR
from MLalgorithms._Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, positive=False):
		self.copy_X = copy_X
		self.n_jobs = n_jobs
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.positive = positive
		self.model = LR(normalize = self.normalize,
			positive = self.positive,
			copy_X = self.copy_X,
			n_jobs = self.n_jobs,
			fit_intercept = self.fit_intercept)

