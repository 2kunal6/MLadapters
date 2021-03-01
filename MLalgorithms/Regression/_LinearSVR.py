
from sklearn.svm import LinearSVR as LSVR
from MLalgorithms._Regression import Regression


class LinearSVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=None, max_iter=1000):
		self.random_state = random_state
		self.dual = dual
		self.max_iter = max_iter
		self.verbose = verbose
		self.tol = tol
		self.epsilon = epsilon
		self.intercept_scaling = intercept_scaling
		self.fit_intercept = fit_intercept
		self.loss = loss
		self.model = LSVR(tol = self.tol,
			epsilon = self.epsilon,
			dual = self.dual,
			random_state = self.random_state,
			max_iter = self.max_iter,
			loss = self.loss,
			verbose = self.verbose,
			fit_intercept = self.fit_intercept,
			intercept_scaling = self.intercept_scaling)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

