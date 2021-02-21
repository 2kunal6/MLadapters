
from sklearn.linear_model import Lasso
from MLalgorithms.Regression import Regression


class LassoRegression(Regression):
	
	def __init__(self, fit_intercept, copy_X, n_jobs, normalize, max_iter):
		self.max_iter = max_iter
		Regression.__init__(self, fit_intercept, copy_X, n_jobs, normalize)
		self.model = Lasso(normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			copy_X = self.copy_X,
			n_jobs = self.n_jobs)

