
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.warm_start = warm_start
		self.precompute = precompute
		self.normalize = normalize
		self.l1_ratio = l1_ratio
		self.positive = positive
		self.random_state = random_state
		self.alpha = alpha
		self.selection = selection
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.max_iter = max_iter
		self.model = ElasticNet(copy_X = self.copy_X,
			normalize = self.normalize,
			precompute = self.precompute,
			l1_ratio = self.l1_ratio,
			max_iter = self.max_iter,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			tol = self.tol,
			random_state = self.random_state,
			alpha = self.alpha,
			positive = self.positive)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			check_input=check_input,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

