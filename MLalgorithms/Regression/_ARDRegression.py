
from sklearn.linear_model import ARDRegression as ARD
from MLalgorithms._Regression import Regression


class ARDRegression(Regression):
	
	def predict(self, X, return_std=False):
		return self.model.predict(return_std=return_std,
			X=X)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.n_iter = n_iter
		self.compute_score = compute_score
		self.lambda_2 = lambda_2
		self.alpha_2 = alpha_2
		self.alpha_1 = alpha_1
		self.lambda_1 = lambda_1
		self.tol = tol
		self.threshold_lambda = threshold_lambda
		self.verbose = verbose
		Regression.__init__(self, normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
		self.model = ARD(compute_score = self.compute_score,
			normalize = self.normalize,
			alpha_1 = self.alpha_1,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			lambda_2 = self.lambda_2,
			tol = self.tol,
			threshold_lambda = self.threshold_lambda,
			n_iter = self.n_iter,
			alpha_2 = self.alpha_2,
			lambda_1 = self.lambda_1,
			verbose = self.verbose)

