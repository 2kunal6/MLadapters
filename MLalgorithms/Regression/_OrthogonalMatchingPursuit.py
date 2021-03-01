
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from MLalgorithms._Regression import Regression


class OrthogonalMatchingPursuit(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=False, precompute='auto'):
		self.tol = tol
		self.n_nonzero_coefs = n_nonzero_coefs
		self.normalize = normalize
		self.fit_intercept = fit_intercept
		self.precompute = precompute
		self.model = OMP(tol = self.tol,
			normalize = self.normalize,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			n_nonzero_coefs = self.n_nonzero_coefs)

	def fit(self, X, y):
		return self.model.fit(X=X,
			y=y)

