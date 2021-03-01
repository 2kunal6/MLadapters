
from sklearn.linear_model import BayesianRidge
from MLalgorithms._Regression import Regression


class BayesianRidgeRegression(Regression):
	
	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.alpha_1 = alpha_1
		self.n_iter = n_iter
		self.compute_score = compute_score
		self.verbose = verbose
		self.lambda_2 = lambda_2
		self.fit_intercept = fit_intercept
		self.alpha_init = alpha_init
		self.copy_X = copy_X
		self.lambda_init = lambda_init
		self.tol = tol
		self.lambda_1 = lambda_1
		self.alpha_2 = alpha_2
		self.normalize = normalize
		self.model = BayesianRidge(alpha_1 = self.alpha_1,
			compute_score = self.compute_score,
			copy_X = self.copy_X,
			lambda_2 = self.lambda_2,
			normalize = self.normalize,
			lambda_init = self.lambda_init,
			fit_intercept = self.fit_intercept,
			alpha_2 = self.alpha_2,
			lambda_1 = self.lambda_1,
			alpha_init = self.alpha_init,
			verbose = self.verbose,
			tol = self.tol,
			n_iter = self.n_iter)

	def predict(self, return_std=False):
		return self.model.predict(return_std=return_std)

