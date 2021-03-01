
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
		self.gamma = gamma
		self.verbose = verbose
		self.kernel = kernel
		self.nu = nu
		self.shrinking = shrinking
		self.cache_size = cache_size
		self.degree = degree
		self.tol = tol
		self.coef0 = coef0
		self.max_iter = max_iter
		self.C = C
		self.model = NSVR(nu = self.nu,
			degree = self.degree,
			verbose = self.verbose,
			coef0 = self.coef0,
			shrinking = self.shrinking,
			kernel = self.kernel,
			gamma = self.gamma,
			max_iter = self.max_iter,
			tol = self.tol,
			C = self.C,
			cache_size = self.cache_size)

