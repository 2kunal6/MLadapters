
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(y=y,
			check_input=check_input,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.selection = selection
		self.normalize = normalize
		self.l1_ratio = l1_ratio
		self.precompute = precompute
		self.random_state = random_state
		self.positive = positive
		self.fit_intercept = fit_intercept
		self.alpha = alpha
		self.copy_X = copy_X
		self.model = ElasticNet(warm_start = self.warm_start,
			max_iter = self.max_iter,
			alpha = self.alpha,
			precompute = self.precompute,
			normalize = self.normalize,
			l1_ratio = self.l1_ratio,
			positive = self.positive,
			selection = self.selection,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			random_state = self.random_state)

