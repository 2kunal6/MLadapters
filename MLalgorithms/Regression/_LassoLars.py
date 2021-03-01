
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.eps = eps
		self.copy_X = copy_X
		self.fit_path = fit_path
		self.precompute = precompute
		self.max_iter = max_iter
		self.jitter = jitter
		self.positive = positive
		self.verbose = verbose
		self.random_state = random_state
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.model = LLR(jitter = self.jitter,
			copy_X = self.copy_X,
			normalize = self.normalize,
			precompute = self.precompute,
			verbose = self.verbose,
			fit_path = self.fit_path,
			max_iter = self.max_iter,
			eps = self.eps,
			fit_intercept = self.fit_intercept,
			random_state = self.random_state,
			alpha = self.alpha,
			positive = self.positive)

	def fit(self, X, y, Xy=None):
		return self.model.fit(X=X,
			y=y,
			Xy=Xy)

	def predict(self, X):
		return self.model.predict(X=X)

