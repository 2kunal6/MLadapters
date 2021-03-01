
from sklearn.linear_model import BayesianRidge
from MLalgorithms._Regression import Regression


class BayesianRidgeRegression(Regression):
	
	def predict(self, return_std=False):
		return self.model.predict(return_std=return_std)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.copy_X = copy_X
		self.alpha_init = alpha_init
		self.verbose = verbose
		self.tol = tol
		self.alpha_2 = alpha_2
		self.n_iter = n_iter
		self.lambda_init = lambda_init
		self.normalize = normalize
		self.compute_score = compute_score
		self.lambda_2 = lambda_2
		self.fit_intercept = fit_intercept
		self.lambda_1 = lambda_1
		self.alpha_1 = alpha_1
		self.model = BayesianRidge(tol = self.tol,
			lambda_1 = self.lambda_1,
			alpha_1 = self.alpha_1,
			n_iter = self.n_iter,
			compute_score = self.compute_score,
			copy_X = self.copy_X,
			alpha_2 = self.alpha_2,
			normalize = self.normalize,
			lambda_2 = self.lambda_2,
			alpha_init = self.alpha_init,
			verbose = self.verbose,
			fit_intercept = self.fit_intercept,
			lambda_init = self.lambda_init)

