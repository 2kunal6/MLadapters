
from MLalgorithms.Regression.LassoRegression import LassoRegression


class LassoTempReg(LassoRegression):
	
	def __init__(self, fit_intercept, copy_X, n_jobs, normalize, max_iter, alpha):
		self.alpha = alpha
		LassoRegression.__init__(self, fit_intercept, copy_X, n_jobs, normalize, max_iter)

