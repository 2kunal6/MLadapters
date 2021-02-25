
from sklearn.linear_model import BayesianRidge
from MLalgorithms._Regression import Regression


class BayesianRidgeRegression(Regression):
	
	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.verbose = verbose
		self.compute_score = compute_score
		self.alpha_init = alpha_init
		self.lambda_1 = lambda_1
		self.n_iter = n_iter
		self.lambda_2 = lambda_2
		self.alpha_1 = alpha_1
		self.lambda_init = lambda_init
		self.alpha_2 = alpha_2
		self.tol = tol
		Regression.__init__(self, normalize=normalize, copy_X=copy_X, fit_intercept=fit_intercept)
		self.model = BayesianRidge(alpha_2 = self.alpha_2,
			n_iter = self.n_iter,
			alpha_init = self.alpha_init,
			lambda_1 = self.lambda_1,
			alpha_1 = self.alpha_1,
			lambda_init = self.lambda_init,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize,
			lambda_2 = self.lambda_2,
			tol = self.tol,
			verbose = self.verbose,
			compute_score = self.compute_score)

	def predict(self, return_std=False, normalize=False, copy_X=True, fit_intercept=True):
		return self.model.predict(return_std=return_std)

