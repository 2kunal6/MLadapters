
from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR as LSVR
from MLalgorithms._Regression import Regression


class LinearSVR(Regression):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, epsilon=0.0, tol=0.0001, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1, dual=True, verbose=0, random_state=None, max_iter=1000, normalize=False, copy_X=True):
		self.epsilon = epsilon
		self.verbose = verbose
		self.max_iter = max_iter
		self.tol = tol
		self.dual = dual
		self.random_state = random_state
		self.loss = loss
		self.intercept_scaling = intercept_scaling
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = LSVR(tol = self.tol,
			copy_X = self.copy_X,
			normalize = self.normalize,
			dual = self.dual,
			intercept_scaling = self.intercept_scaling,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			verbose = self.verbose,
			max_iter = self.max_iter,
			epsilon = self.epsilon,
			loss = self.loss)

	def predict(self, X):
		return self.model.predict(X=X)

