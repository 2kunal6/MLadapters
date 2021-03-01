
from sklearn.linear_model import BayesianRidge
from MLalgorithms._Regression import Regression


class BayesianRidgeRegression(Regression):
	
	def predict(self, return_std=False):
		return self.model.predict(return_std=return_std)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_x=True, verbose=False):
		self.lambda_init = lambda_init
		self.alpha_init = alpha_init
		self.alpha_1 = alpha_1
		self.tol = tol
		self.alpha_2 = alpha_2
		self.lambda_2 = lambda_2
		self.n_iter = n_iter
		self.verbose = verbose
		self.compute_score = compute_score
		self.lambda_1 = lambda_1
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_x=copy_x)
		self.model = BayesianRidge(lambda_init = self.lambda_init,
			copy_x = self.copy_x,
			alpha_init = self.alpha_init,
			tol = self.tol,
			compute_score = self.compute_score,
			n_iter = self.n_iter,
			verbose = self.verbose,
			lambda_1 = self.lambda_1,
			normalize = self.normalize,
			lambda_2 = self.lambda_2,
			alpha_1 = self.alpha_1,
			alpha_2 = self.alpha_2,
			fit_intercept = self.fit_intercept)

