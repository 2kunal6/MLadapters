
from sklearn.svm import LinearSVR as LSVR
from MLalgorithms._Regression import Regression


class LinearSVR(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=None, max_iter=1000):
		self.dual = dual
		self.verbose = verbose
		self.tol = tol
		self.intercept_scaling = intercept_scaling
		self.loss = loss
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.random_state = random_state
		self.epsilon = epsilon
		self.model = LSVR(loss = self.loss,
			verbose = self.verbose,
			dual = self.dual,
			max_iter = self.max_iter,
			intercept_scaling = self.intercept_scaling,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			random_state = self.random_state,
			epsilon = self.epsilon)

