
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
		self.coef0 = coef0
		self.cache_size = cache_size
		self.degree = degree
		self.max_iter = max_iter
		self.C = C
		self.kernel = kernel
		self.shrinking = shrinking
		self.verbose = verbose
		self.nu = nu
		self.tol = tol
		self.gamma = gamma
		self.model = NSVR(nu = self.nu,
			shrinking = self.shrinking,
			cache_size = self.cache_size,
			verbose = self.verbose,
			max_iter = self.max_iter,
			gamma = self.gamma,
			tol = self.tol,
			coef0 = self.coef0,
			C = self.C,
			degree = self.degree,
			kernel = self.kernel)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

