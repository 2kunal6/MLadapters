
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.precompute = precompute
		self.l1_ratio = l1_ratio
		self.random_state = random_state
		self.selection = selection
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.tol = tol
		self.copy_X = copy_X
		self.alpha = alpha
		self.positive = positive
		self.normalize = normalize
		self.warm_start = warm_start
		self.model = ElasticNet(normalize = self.normalize,
			copy_X = self.copy_X,
			random_state = self.random_state,
			precompute = self.precompute,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			selection = self.selection,
			warm_start = self.warm_start,
			l1_ratio = self.l1_ratio,
			positive = self.positive,
			tol = self.tol,
			alpha = self.alpha)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(y=y,
			check_input=check_input,
			sample_weight=sample_weight,
			X=X)

