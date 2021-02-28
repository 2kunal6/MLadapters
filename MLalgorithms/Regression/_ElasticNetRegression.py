
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			check_input=check_input,
			X=X)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_x=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.precompute = precompute
		self.selection = selection
		self.max_iter = max_iter
		self.l1_ratio = l1_ratio
		self.positive = positive
		self.alpha = alpha
		self.warm_start = warm_start
		self.tol = tol
		self.random_state = random_state
		Regression.__init__(self, copy_x=copy_x, normalize=normalize, fit_intercept=fit_intercept)
		self.model = ElasticNet(copy_x = self.copy_x,
			tol = self.tol,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			alpha = self.alpha,
			normalize = self.normalize,
			positive = self.positive,
			l1_ratio = self.l1_ratio,
			precompute = self.precompute,
			selection = self.selection,
			max_iter = self.max_iter)

