
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y,
			check_input=check_input)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_x=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.max_iter = max_iter
		self.selection = selection
		self.precompute = precompute
		self.alpha = alpha
		self.positive = positive
		self.l1_ratio = l1_ratio
		self.warm_start = warm_start
		self.tol = tol
		self.random_state = random_state
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_x=copy_x)
		self.model = ElasticNet(warm_start = self.warm_start,
			copy_x = self.copy_x,
			tol = self.tol,
			precompute = self.precompute,
			alpha = self.alpha,
			selection = self.selection,
			normalize = self.normalize,
			positive = self.positive,
			random_state = self.random_state,
			l1_ratio = self.l1_ratio,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept)

