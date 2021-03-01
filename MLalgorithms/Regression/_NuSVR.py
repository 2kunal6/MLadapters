
from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVR as NSVR
from MLalgorithms._Regression import Regression


class NuSVR(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1, fit_intercept=True, normalize=False, copy_X=True):
		self.shrinking = shrinking
		self.verbose = verbose
		self.C = C
		self.tol = tol
		self.degree = degree
		self.cache_size = cache_size
		self.coef0 = coef0
		self.nu = nu
		self.kernel = kernel
		self.max_iter = max_iter
		self.gamma = gamma
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = NSVR(nu = self.nu,
			shrinking = self.shrinking,
			kernel = self.kernel,
			copy_X = self.copy_X,
			C = self.C,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			gamma = self.gamma,
			degree = self.degree,
			verbose = self.verbose,
			max_iter = self.max_iter,
			tol = self.tol,
			coef0 = self.coef0,
			cache_size = self.cache_size)

	def predict(self, X):
		return self.model.predict(X=X)

