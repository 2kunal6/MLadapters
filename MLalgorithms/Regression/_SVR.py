
from sklearn.svm import SVR as SVRRegression
from MLalgorithms._Regression import Regression


class SVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
		self.tol = tol
		self.shrinking = shrinking
		self.C = C
		self.verbose = verbose
		self.degree = degree
		self.epsilon = epsilon
		self.coef0 = coef0
		self.kernel = kernel
		self.cache_size = cache_size
		self.gamma = gamma
		self.max_iter = max_iter
		self.model = SVRRegression(shrinking = self.shrinking,
			epsilon = self.epsilon,
			kernel = self.kernel,
			degree = self.degree,
			max_iter = self.max_iter,
			verbose = self.verbose,
			tol = self.tol,
			gamma = self.gamma,
			C = self.C,
			cache_size = self.cache_size,
			coef0 = self.coef0)

