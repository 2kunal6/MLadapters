
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from MLalgorithms._Regression import Regression


class OrthogonalMatchingPursuit(Regression):
	
	def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=False, precompute='auto'):
		self.precompute = precompute
		self.normalize = normalize
		self.n_nonzero_coefs = n_nonzero_coefs
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.model = OMP(n_nonzero_coefs = self.n_nonzero_coefs,
			normalize = self.normalize,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			tol = self.tol)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

