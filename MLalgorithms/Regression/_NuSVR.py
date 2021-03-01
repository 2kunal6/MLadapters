
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1):
		self.kernel = kernel
		self.verbose = verbose
		self.degree = degree
		self.C = C
		self.max_iter = max_iter
		self.cache_size = cache_size
		self.gamma = gamma
		self.shrinking = shrinking
		self.nu = nu
		self.tol = tol
		self.coef0 = coef0
		self.model = NSVR(cache_size = self.cache_size,
			max_iter = self.max_iter,
			nu = self.nu,
			shrinking = self.shrinking,
			verbose = self.verbose,
			gamma = self.gamma,
			C = self.C,
			kernel = self.kernel,
			degree = self.degree,
			tol = self.tol,
			coef0 = self.coef0)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

