
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from MLalgorithms._Regression import Regression


class OrthogonalMatchingPursuit(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=False, precompute='auto'):
		self.tol = tol
		self.normalize = normalize
		self.fit_intercept = fit_intercept
		self.precompute = precompute
		self.n_nonzero_coefs = n_nonzero_coefs
		self.model = OMP(n_nonzero_coefs = self.n_nonzero_coefs,
			precompute = self.precompute,
			normalize = self.normalize,
			tol = self.tol,
			fit_intercept = self.fit_intercept)

