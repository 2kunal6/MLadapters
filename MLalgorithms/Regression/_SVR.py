
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR as SVRRegression
from MLalgorithms._Regression import Regression


class SVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
		self.tol = tol
		self.max_iter = max_iter
		self.cache_size = cache_size
		self.gamma = gamma
		self.kernel = kernel
		self.epsilon = epsilon
		self.degree = degree
		self.C = C
		self.coef0 = coef0
		self.verbose = verbose
		self.shrinking = shrinking
		self.model = SVRRegression(coef0 = self.coef0,
			tol = self.tol,
			cache_size = self.cache_size,
			degree = self.degree,
			gamma = self.gamma,
			max_iter = self.max_iter,
			shrinking = self.shrinking,
			verbose = self.verbose,
			epsilon = self.epsilon,
			C = self.C,
			kernel = self.kernel)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

