
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, Xy=None):
		return self.model.fit(Xy=Xy,
			y=y,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.max_iter = max_iter
		self.verbose = verbose
		self.eps = eps
		self.normalize = normalize
		self.jitter = jitter
		self.precompute = precompute
		self.random_state = random_state
		self.fit_path = fit_path
		self.positive = positive
		self.fit_intercept = fit_intercept
		self.alpha = alpha
		self.copy_X = copy_X
		self.model = LLR(fit_path = self.fit_path,
			max_iter = self.max_iter,
			alpha = self.alpha,
			precompute = self.precompute,
			jitter = self.jitter,
			normalize = self.normalize,
			positive = self.positive,
			eps = self.eps,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			verbose = self.verbose,
			random_state = self.random_state)

