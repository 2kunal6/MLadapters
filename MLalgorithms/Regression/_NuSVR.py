
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
		self.tol = tol
		self.shrinking = shrinking
		self.C = C
		self.verbose = verbose
		self.degree = degree
		self.nu = nu
		self.coef0 = coef0
		self.kernel = kernel
		self.cache_size = cache_size
		self.gamma = gamma
		self.max_iter = max_iter
		self.model = NSVR(shrinking = self.shrinking,
			kernel = self.kernel,
			degree = self.degree,
			max_iter = self.max_iter,
			verbose = self.verbose,
			nu = self.nu,
			tol = self.tol,
			gamma = self.gamma,
			C = self.C,
			cache_size = self.cache_size,
			coef0 = self.coef0)

