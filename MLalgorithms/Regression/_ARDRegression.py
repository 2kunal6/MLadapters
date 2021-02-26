
from sklearn.linear_model import ARDRegression as ARD
from MLalgorithms._Regression import Regression


class ARDRegression(Regression):
	
	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.lambda_1 = lambda_1
		self.lambda_2 = lambda_2
		self.compute_score = compute_score
		self.verbose = verbose
		self.alpha_2 = alpha_2
		self.alpha_1 = alpha_1
		self.threshold_lambda = threshold_lambda
		self.n_iter = n_iter
		self.tol = tol
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X)
		self.model = ARD(tol = self.tol,
			alpha_2 = self.alpha_2,
			lambda_1 = self.lambda_1,
			fit_intercept = self.fit_intercept,
			verbose = self.verbose,
			threshold_lambda = self.threshold_lambda,
			alpha_1 = self.alpha_1,
			n_iter = self.n_iter,
			normalize = self.normalize,
			compute_score = self.compute_score,
			copy_X = self.copy_X,
			lambda_2 = self.lambda_2)

	def predict(self, return_std=False):
		return self.model.predict(return_std=return_std)

