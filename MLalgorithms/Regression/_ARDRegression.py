
from sklearn.linear_model import ARDRegression as ARD
from MLalgorithms._Regression import Regression


class ARDRegression(Regression):
	
	def predict(self, X, return_std=False):
		return self.model.predict(X=X,
			return_std=return_std)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000, fit_intercept=True, normalize=False, copy_x=True, verbose=False):
		self.alpha_1 = alpha_1
		self.tol = tol
		self.alpha_2 = alpha_2
		self.lambda_2 = lambda_2
		self.n_iter = n_iter
		self.threshold_lambda = threshold_lambda
		self.verbose = verbose
		self.compute_score = compute_score
		self.lambda_1 = lambda_1
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_x=copy_x)
		self.model = ARD(copy_x = self.copy_x,
			tol = self.tol,
			compute_score = self.compute_score,
			n_iter = self.n_iter,
			verbose = self.verbose,
			lambda_1 = self.lambda_1,
			normalize = self.normalize,
			lambda_2 = self.lambda_2,
			alpha_1 = self.alpha_1,
			alpha_2 = self.alpha_2,
			threshold_lambda = self.threshold_lambda,
			fit_intercept = self.fit_intercept)

