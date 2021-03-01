
from sklearn.linear_model import ARDRegression as ARD
from MLalgorithms._Regression import Regression


class ARDRegression(Regression):
	
	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.lambda_2 = lambda_2
		self.verbose = verbose
		self.tol = tol
		self.alpha_1 = alpha_1
		self.threshold_lambda = threshold_lambda
		self.alpha_2 = alpha_2
		self.n_iter = n_iter
		self.compute_score = compute_score
		self.lambda_1 = lambda_1
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = ARD(n_iter = self.n_iter,
			lambda_2 = self.lambda_2,
			copy_X = self.copy_X,
			normalize = self.normalize,
			alpha_2 = self.alpha_2,
			alpha_1 = self.alpha_1,
			fit_intercept = self.fit_intercept,
			compute_score = self.compute_score,
			lambda_1 = self.lambda_1,
			threshold_lambda = self.threshold_lambda,
			verbose = self.verbose,
			tol = self.tol)

	def predict(self, X, return_std=False):
		return self.model.predict(return_std=return_std,
			X=X)

