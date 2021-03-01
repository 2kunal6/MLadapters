
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.copy_X = copy_X
		self.random_state = random_state
		self.warm_start = warm_start
		self.fit_intercept = fit_intercept
		self.positive = positive
		self.max_iter = max_iter
		self.tol = tol
		self.selection = selection
		self.normalize = normalize
		self.alpha = alpha
		self.l1_ratio = l1_ratio
		self.precompute = precompute
		self.model = ElasticNet(tol = self.tol,
			precompute = self.precompute,
			copy_X = self.copy_X,
			normalize = self.normalize,
			l1_ratio = self.l1_ratio,
			random_state = self.random_state,
			positive = self.positive,
			max_iter = self.max_iter,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(X=X,
			check_input=check_input,
			sample_weight=sample_weight,
			y=y)

