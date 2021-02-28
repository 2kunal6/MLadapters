
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_x=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.selection = selection
		self.tol = tol
		self.random_state = random_state
		self.positive = positive
		self.precompute = precompute
		self.l1_ratio = l1_ratio
		self.alpha = alpha
		self.max_iter = max_iter
		self.warm_start = warm_start
		Regression.__init__(self, normalize=normalize, copy_x=copy_x, fit_intercept=fit_intercept)
		self.model = ElasticNet(max_iter = self.max_iter,
			precompute = self.precompute,
			positive = self.positive,
			random_state = self.random_state,
			l1_ratio = self.l1_ratio,
			alpha = self.alpha,
			selection = self.selection,
			warm_start = self.warm_start,
			normalize = self.normalize,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			copy_x = self.copy_x)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(X=X,
			y=y,
			check_input=check_input,
			sample_weight=sample_weight)

