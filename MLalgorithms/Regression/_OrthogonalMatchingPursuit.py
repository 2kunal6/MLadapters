
from sklearn.datasets import make_regression
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from MLalgorithms._Regression import Regression


class OrthogonalMatchingPursuit(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=False, precompute='auto'):
		self.n_nonzero_coefs = n_nonzero_coefs
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.precompute = precompute
		self.normalize = normalize
		self.model = OMP(tol = self.tol,
			normalize = self.normalize,
			n_nonzero_coefs = self.n_nonzero_coefs,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

