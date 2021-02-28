
from sklearn.linear_model import BayesianRidge
from MLalgorithms._Regression import Regression


class BayesianRidgeRegression(Regression):
	
	def predict(self, return_std=False):
		return self.model.predict(return_std=return_std)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_x=True, verbose=False):
		self.alpha_1 = alpha_1
		self.lambda_init = lambda_init
		self.lambda_1 = lambda_1
		self.lambda_2 = lambda_2
		self.alpha_init = alpha_init
		self.compute_score = compute_score
		self.alpha_2 = alpha_2
		self.tol = tol
		self.verbose = verbose
		self.n_iter = n_iter
		Regression.__init__(self, normalize=normalize, copy_x=copy_x, fit_intercept=fit_intercept)
		self.model = BayesianRidge(alpha_1 = self.alpha_1,
			alpha_init = self.alpha_init,
			alpha_2 = self.alpha_2,
			n_iter = self.n_iter,
			compute_score = self.compute_score,
			copy_x = self.copy_x,
			lambda_1 = self.lambda_1,
			normalize = self.normalize,
			tol = self.tol,
			lambda_2 = self.lambda_2,
			lambda_init = self.lambda_init,
			fit_intercept = self.fit_intercept,
			verbose = self.verbose)

