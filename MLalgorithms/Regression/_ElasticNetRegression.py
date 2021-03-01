
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
		self.normalize = normalize
		self.tol = tol
		self.copy_X = copy_X
		self.selection = selection
		self.warm_start = warm_start
		self.l1_ratio = l1_ratio
		self.alpha = alpha
		self.random_state = random_state
		self.positive = positive
		self.precompute = precompute
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.model = ElasticNet(l1_ratio = self.l1_ratio,
			copy_X = self.copy_X,
			positive = self.positive,
			max_iter = self.max_iter,
			random_state = self.random_state,
			alpha = self.alpha,
			precompute = self.precompute,
			tol = self.tol,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			selection = self.selection,
			warm_start = self.warm_start)

