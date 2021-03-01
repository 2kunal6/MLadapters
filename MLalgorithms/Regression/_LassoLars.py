
from sklearn.linear_model import LassoLars as LLR
from MLalgorithms._Regression import Regression


class LassoLars(Regression):
	
	def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=False, precompute='auto', max_iter=500, eps=2.220446e-16, copy_X=True, fit_path=True, positive=False, jitter=None, random_state=None):
		self.precompute = precompute
		self.random_state = random_state
		self.eps = eps
		self.verbose = verbose
		self.fit_intercept = fit_intercept
		self.jitter = jitter
		self.max_iter = max_iter
		self.copy_X = copy_X
		self.alpha = alpha
		self.positive = positive
		self.fit_path = fit_path
		self.normalize = normalize
		self.model = LLR(normalize = self.normalize,
			copy_X = self.copy_X,
			random_state = self.random_state,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			eps = self.eps,
			max_iter = self.max_iter,
			fit_path = self.fit_path,
			jitter = self.jitter,
			verbose = self.verbose,
			positive = self.positive,
			alpha = self.alpha)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, Xy=None):
		return self.model.fit(y=y,
			X=X,
			Xy=Xy)

