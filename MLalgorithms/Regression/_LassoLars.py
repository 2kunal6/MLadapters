
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def fit(self, X, y, Xy=None):
		return self.model.fit(Xy=Xy,
			y=y,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.alpha = alpha
		self.verbose = verbose
		self.positive = positive
		self.eps = eps
		self.jitter = jitter
		self.max_iter = max_iter
		self.random_state = random_state
		self.precompute = precompute
		self.fit_path = fit_path
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = LLR(max_iter = self.max_iter,
			alpha = self.alpha,
			copy_X = self.copy_X,
			normalize = self.normalize,
			positive = self.positive,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			jitter = self.jitter,
			verbose = self.verbose,
			eps = self.eps,
			fit_path = self.fit_path)

	def predict(self, X):
		return self.model.predict(X=X)

