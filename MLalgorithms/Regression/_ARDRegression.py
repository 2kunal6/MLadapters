
from sklearn.linear_model import ARDRegression as ARD
from MLalgorithms._Regression import Regression


class ARDRegression(Regression):
	
	def predict(self, X, return_std=False):
		return self.model.predict(X=X,
			return_std=return_std)

	def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000, fit_intercept=True, normalize=False, copy_x=True, verbose=False):
		self.n_iter = n_iter
		self.alpha_1 = alpha_1
		self.verbose = verbose
		self.lambda_1 = lambda_1
		self.tol = tol
		self.compute_score = compute_score
		self.threshold_lambda = threshold_lambda
		self.lambda_2 = lambda_2
		self.alpha_2 = alpha_2
		Regression.__init__(self, copy_x=copy_x, normalize=normalize, fit_intercept=fit_intercept)
		self.model = ARD(lambda_1 = self.lambda_1,
			copy_x = self.copy_x,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			verbose = self.verbose,
			alpha_1 = self.alpha_1,
			alpha_2 = self.alpha_2,
			normalize = self.normalize,
			compute_score = self.compute_score,
			lambda_2 = self.lambda_2,
			n_iter = self.n_iter,
			threshold_lambda = self.threshold_lambda)

