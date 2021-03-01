
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
		self.alpha = alpha
		self.eps = eps
		self.verbose = verbose
		self.precompute = precompute
		self.jitter = jitter
		self.fit_intercept = fit_intercept
		self.positive = positive
		self.random_state = random_state
		self.copy_X = copy_X
		self.fit_path = fit_path
		self.normalize = normalize
		self.max_iter = max_iter
		self.model = LLR(precompute = self.precompute,
			normalize = self.normalize,
			verbose = self.verbose,
			alpha = self.alpha,
			fit_path = self.fit_path,
			max_iter = self.max_iter,
			jitter = self.jitter,
			fit_intercept = self.fit_intercept,
			random_state = self.random_state,
			copy_X = self.copy_X,
			eps = self.eps,
			positive = self.positive)

