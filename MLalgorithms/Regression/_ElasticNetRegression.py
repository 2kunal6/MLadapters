
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			check_input=check_input,
			X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.warm_start = warm_start
		self.alpha = alpha
		self.tol = tol
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.positive = positive
		self.copy_X = copy_X
		self.normalize = normalize
		self.precompute = precompute
		self.l1_ratio = l1_ratio
		self.random_state = random_state
		self.selection = selection
		self.model = ElasticNet(precompute = self.precompute,
			l1_ratio = self.l1_ratio,
			normalize = self.normalize,
			selection = self.selection,
			warm_start = self.warm_start,
			alpha = self.alpha,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			random_state = self.random_state,
			copy_X = self.copy_X,
			positive = self.positive)

