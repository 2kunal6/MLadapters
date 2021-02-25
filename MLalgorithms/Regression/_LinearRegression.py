
from sklearn.linear_model import LinearRegression as LR
from MLalgorithms._Regression import Regression


class LinearRegression(Regression):
	
	def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, positive=False):
		self.positive = positive
		self.n_jobs = n_jobs
		Regression.__init__(self, normalize=normalize, copy_X=copy_X, fit_intercept=fit_intercept)
		self.model = LR(fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			normalize = self.normalize,
			positive = self.positive,
			n_jobs = self.n_jobs)

