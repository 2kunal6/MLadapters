
from sklearn.svm import LinearSVR as LSVR
from MLalgorithms._Regression import Regression


class LinearSVR(Regression):
	
	def __init__(self, epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=None, max_iter=1000):
		self.verbose = verbose
		self.dual = dual
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.loss = loss
		self.tol = tol
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.intercept_scaling = intercept_scaling
		self.model = LSVR(epsilon = self.epsilon,
			dual = self.dual,
			verbose = self.verbose,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			loss = self.loss,
			tol = self.tol,
			random_state = self.random_state,
			intercept_scaling = self.intercept_scaling)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

