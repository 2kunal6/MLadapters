
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
		self.cache_size = cache_size
		self.verbose = verbose
		self.tol = tol
		self.coef0 = coef0
		self.shrinking = shrinking
		self.gamma = gamma
		self.kernel = kernel
		self.max_iter = max_iter
		self.C = C
		self.nu = nu
		self.degree = degree
		self.model = NSVR(C = self.C,
			degree = self.degree,
			shrinking = self.shrinking,
			max_iter = self.max_iter,
			coef0 = self.coef0,
			tol = self.tol,
			nu = self.nu,
			verbose = self.verbose,
			cache_size = self.cache_size,
			gamma = self.gamma,
			kernel = self.kernel)

