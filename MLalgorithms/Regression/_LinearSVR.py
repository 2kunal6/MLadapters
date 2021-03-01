
from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR as LSVR
from MLalgorithms._Regression import Regression


class LinearSVR(Regression):
	
	def __init__(self, epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=None, max_iter=1000):
		self.random_state = random_state
		self.loss = loss
		self.verbose = verbose
		self.fit_intercept = fit_intercept
		self.dual = dual
		self.tol = tol
		self.max_iter = max_iter
		self.epsilon = epsilon
		self.intercept_scaling = intercept_scaling
		self.model = LSVR(random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			epsilon = self.epsilon,
			dual = self.dual,
			intercept_scaling = self.intercept_scaling,
			verbose = self.verbose,
			tol = self.tol,
			loss = self.loss)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

