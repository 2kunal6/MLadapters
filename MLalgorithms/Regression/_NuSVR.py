
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
		self.verbose = verbose
		self.degree = degree
		self.tol = tol
		self.cache_size = cache_size
		self.C = C
		self.max_iter = max_iter
		self.kernel = kernel
		self.gamma = gamma
		self.shrinking = shrinking
		self.coef0 = coef0
		self.nu = nu
		self.model = NSVR(degree = self.degree,
			max_iter = self.max_iter,
			coef0 = self.coef0,
			cache_size = self.cache_size,
			tol = self.tol,
			gamma = self.gamma,
			nu = self.nu,
			C = self.C,
			shrinking = self.shrinking,
			verbose = self.verbose,
			kernel = self.kernel)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

