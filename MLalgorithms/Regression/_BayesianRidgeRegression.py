
from sklearn.linear_model import BayesianRidge
from MLalgorithms._Regression import Regression


class BayesianRidgeRegression(Regression):
	
	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.lambda_2 = lambda_2
		self.lambda_init = lambda_init
		self.verbose = verbose
		self.alpha_init = alpha_init
		self.tol = tol
		self.alpha_1 = alpha_1
		self.alpha_2 = alpha_2
		self.n_iter = n_iter
		self.compute_score = compute_score
		self.lambda_1 = lambda_1
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = BayesianRidge(n_iter = self.n_iter,
			lambda_2 = self.lambda_2,
			copy_X = self.copy_X,
			normalize = self.normalize,
			alpha_2 = self.alpha_2,
			lambda_init = self.lambda_init,
			alpha_1 = self.alpha_1,
			alpha_init = self.alpha_init,
			compute_score = self.compute_score,
			fit_intercept = self.fit_intercept,
			lambda_1 = self.lambda_1,
			verbose = self.verbose,
			tol = self.tol)

	def predict(self, return_std=False):
		return self.model.predict(return_std=return_std)

