
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
		self.cache_size = cache_size
		self.verbose = verbose
		self.tol = tol
		self.coef0 = coef0
		self.epsilon = epsilon
		self.shrinking = shrinking
		self.gamma = gamma
		self.kernel = kernel
		self.max_iter = max_iter
		self.C = C
		self.degree = degree
		self.model = SVRRegression(C = self.C,
			degree = self.degree,
			shrinking = self.shrinking,
			epsilon = self.epsilon,
			coef0 = self.coef0,
			max_iter = self.max_iter,
			tol = self.tol,
			verbose = self.verbose,
			cache_size = self.cache_size,
			gamma = self.gamma,
			kernel = self.kernel)

