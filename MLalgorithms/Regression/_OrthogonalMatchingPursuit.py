
from sklearn.datasets import make_regression
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from MLalgorithms._Regression import Regression


class OrthogonalMatchingPursuit(Regression):
	
	def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=False, precompute='auto'):
		self.n_nonzero_coefs = n_nonzero_coefs
		self.precompute = precompute
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.model = OMP(normalize = self.normalize,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			n_nonzero_coefs = self.n_nonzero_coefs,
			tol = self.tol)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

