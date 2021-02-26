
from sklearn.linear_model import BayesianRidge
from MLalgorithms._Regression import Regression


class BayesianRidgeRegression(Regression):
	
	def predict(self, return_std=False):
		return self.model.predict(return_std=return_std)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.alpha_init = alpha_init
		self.n_iter = n_iter
		self.compute_score = compute_score
		self.lambda_init = lambda_init
		self.lambda_2 = lambda_2
		self.alpha_2 = alpha_2
		self.alpha_1 = alpha_1
		self.lambda_1 = lambda_1
		self.tol = tol
		self.verbose = verbose
		Regression.__init__(self, normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
		self.model = BayesianRidge(compute_score = self.compute_score,
			normalize = self.normalize,
			alpha_1 = self.alpha_1,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			lambda_2 = self.lambda_2,
			lambda_1 = self.lambda_1,
			tol = self.tol,
			alpha_init = self.alpha_init,
			n_iter = self.n_iter,
			alpha_2 = self.alpha_2,
			lambda_init = self.lambda_init,
			verbose = self.verbose)

