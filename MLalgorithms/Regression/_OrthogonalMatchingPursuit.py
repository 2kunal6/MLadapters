
from sklearn.datasets import make_regression
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from MLalgorithms._Regression import Regression


class OrthogonalMatchingPursuit(Regression):
	
	def fit(self, X, y):
		return self.model.fit(y=y,
			X=X)

	def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=False, precompute='auto', copy_X=True):
		self.n_nonzero_coefs = n_nonzero_coefs
		self.precompute = precompute
		self.tol = tol
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = OMP(copy_X = self.copy_X,
			normalize = self.normalize,
			n_nonzero_coefs = self.n_nonzero_coefs,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			tol = self.tol)

	def predict(self, X):
		return self.model.predict(X=X)

