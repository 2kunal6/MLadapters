
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, Xy=None):
		return self.model.fit(y=y,
			X=X,
			Xy=Xy)

	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.max_iter = max_iter
		self.normalize = normalize
		self.copy_X = copy_X
		self.jitter = jitter
		self.precompute = precompute
		self.alpha = alpha
		self.verbose = verbose
		self.eps = eps
		self.random_state = random_state
		self.positive = positive
		self.fit_path = fit_path
		self.fit_intercept = fit_intercept
		self.model = LLR(copy_X = self.copy_X,
			positive = self.positive,
			max_iter = self.max_iter,
			verbose = self.verbose,
			random_state = self.random_state,
			alpha = self.alpha,
			fit_path = self.fit_path,
			precompute = self.precompute,
			eps = self.eps,
			jitter = self.jitter,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize)

