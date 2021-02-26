
from sklearn.linear_model import ARDRegression as ARD
from MLalgorithms._Regression import Regression


class ARDRegression(Regression):
	
	def predict(self, return_std=False, normalize=False, copy_X=True, fit_intercept=True):
		return self.model.predict(return_std=return_std)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
		self.threshold_lambda = threshold_lambda
		self.n_iter = n_iter
		self.tol = tol
		self.verbose = verbose
		self.alpha_2 = alpha_2
		self.lambda_2 = lambda_2
		self.compute_score = compute_score
		self.lambda_1 = lambda_1
		self.alpha_1 = alpha_1
		Regression.__init__(self, normalize=normalize, copy_X=copy_X, fit_intercept=fit_intercept)
		self.model = ARD(copy_X = self.copy_X,
			verbose = self.verbose,
			alpha_2 = self.alpha_2,
			n_iter = self.n_iter,
			normalize = self.normalize,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			compute_score = self.compute_score,
			threshold_lambda = self.threshold_lambda,
			lambda_1 = self.lambda_1,
			alpha_1 = self.alpha_1,
			lambda_2 = self.lambda_2)

