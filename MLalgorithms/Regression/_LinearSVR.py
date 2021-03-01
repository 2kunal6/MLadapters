
from sklearn.svm import LinearSVR as LSVR
from MLalgorithms._Regression import Regression


class LinearSVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=None, max_iter=1000):
		self.tol = tol
		self.verbose = verbose
		self.loss = loss
		self.intercept_scaling = intercept_scaling
		self.random_state = random_state
		self.dual = dual
		self.epsilon = epsilon
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.model = LSVR(loss = self.loss,
			dual = self.dual,
			epsilon = self.epsilon,
			max_iter = self.max_iter,
			verbose = self.verbose,
			random_state = self.random_state,
			intercept_scaling = self.intercept_scaling,
			tol = self.tol,
			fit_intercept = self.fit_intercept)

