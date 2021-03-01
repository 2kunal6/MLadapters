
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.warm_start = warm_start
		self.selection = selection
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.precompute = precompute
		self.tol = tol
		self.random_state = random_state
		self.l1_ratio = l1_ratio
		self.max_iter = max_iter
		self.normalize = normalize
		self.positive = positive
		self.model = ElasticNet(selection = self.selection,
			random_state = self.random_state,
			tol = self.tol,
			normalize = self.normalize,
			alpha = self.alpha,
			positive = self.positive,
			warm_start = self.warm_start,
			l1_ratio = self.l1_ratio,
			copy_X = self.copy_X,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			max_iter = self.max_iter)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(check_input=check_input,
			sample_weight=sample_weight,
			X=X,
			y=y)
