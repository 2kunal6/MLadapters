
from sklearn.linear_model import ARDRegression as ARD
from MLalgorithms._Regression import Regression


class ARDRegression(Regression):
	
	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.copy_X = copy_X
		self.verbose = verbose
		self.tol = tol
		self.compute_score = compute_score
		self.normalize = normalize
		self.n_iter = n_iter
		self.alpha_1 = alpha_1
		self.lambda_2 = lambda_2
		self.lambda_1 = lambda_1
		self.alpha_2 = alpha_2
		self.threshold_lambda = threshold_lambda
		self.fit_intercept = fit_intercept
		self.model = ARD(normalize = self.normalize,
			lambda_2 = self.lambda_2,
			copy_X = self.copy_X,
			compute_score = self.compute_score,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			n_iter = self.n_iter,
			verbose = self.verbose,
			alpha_2 = self.alpha_2,
			threshold_lambda = self.threshold_lambda,
			alpha_1 = self.alpha_1,
			lambda_1 = self.lambda_1)

	def predict(self, X, return_std=False):
		return self.model.predict(return_std=return_std,
			X=X)

