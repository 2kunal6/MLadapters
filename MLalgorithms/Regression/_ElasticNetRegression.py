
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.precompute = precompute
		self.l1_ratio = l1_ratio
		self.max_iter = max_iter
		self.normalize = normalize
		self.positive = positive
		self.random_state = random_state
		self.alpha = alpha
		self.tol = tol
		self.selection = selection
		self.warm_start = warm_start
		self.fit_intercept = fit_intercept
		self.model = ElasticNet(max_iter = self.max_iter,
			warm_start = self.warm_start,
			normalize = self.normalize,
			alpha = self.alpha,
			copy_X = self.copy_X,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			tol = self.tol,
			l1_ratio = self.l1_ratio,
			random_state = self.random_state,
			positive = self.positive)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			check_input=check_input,
			y=y,
			X=X)

