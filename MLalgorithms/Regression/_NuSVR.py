
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
		self.C = C
		self.shrinking = shrinking
		self.verbose = verbose
		self.max_iter = max_iter
		self.tol = tol
		self.cache_size = cache_size
		self.nu = nu
		self.coef0 = coef0
		self.degree = degree
		self.kernel = kernel
		self.gamma = gamma
		self.model = NSVR(tol = self.tol,
			coef0 = self.coef0,
			C = self.C,
			degree = self.degree,
			gamma = self.gamma,
			shrinking = self.shrinking,
			max_iter = self.max_iter,
			kernel = self.kernel,
			verbose = self.verbose,
			nu = self.nu,
			cache_size = self.cache_size)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

