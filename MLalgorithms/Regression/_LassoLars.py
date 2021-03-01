
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.copy_X = copy_X
		self.verbose = verbose
		self.eps = eps
		self.fit_path = fit_path
		self.normalize = normalize
		self.positive = positive
		self.random_state = random_state
		self.alpha = alpha
		self.max_iter = max_iter
		self.precompute = precompute
		self.jitter = jitter
		self.fit_intercept = fit_intercept
		self.model = LLR(max_iter = self.max_iter,
			eps = self.eps,
			normalize = self.normalize,
			alpha = self.alpha,
			fit_path = self.fit_path,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			random_state = self.random_state,
			positive = self.positive,
			verbose = self.verbose,
			jitter = self.jitter)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, Xy=None):
		return self.model.fit(Xy=Xy,
			y=y,
			X=X)

