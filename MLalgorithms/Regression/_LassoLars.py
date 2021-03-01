
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.copy_X = copy_X
		self.max_iter = max_iter
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.eps = eps
		self.random_state = random_state
		self.precompute = precompute
		self.jitter = jitter
		self.normalize = normalize
		self.verbose = verbose
		self.fit_path = fit_path
		self.positive = positive
		self.model = LLR(random_state = self.random_state,
			normalize = self.normalize,
			fit_path = self.fit_path,
			alpha = self.alpha,
			positive = self.positive,
			copy_X = self.copy_X,
			eps = self.eps,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			precompute = self.precompute,
			verbose = self.verbose,
			jitter = self.jitter)

	def fit(self, X, y, Xy=None):
		return self.model.fit(y=y,
			Xy=Xy,
			X=X)

