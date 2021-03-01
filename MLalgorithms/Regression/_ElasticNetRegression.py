
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_x=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.tol = tol
		self.warm_start = warm_start
		self.precompute = precompute
		self.max_iter = max_iter
		self.selection = selection
		self.random_state = random_state
		self.positive = positive
		self.alpha = alpha
		self.l1_ratio = l1_ratio
		Regression.__init__(self, copy_x=copy_x, fit_intercept=fit_intercept, normalize=normalize)
		self.model = ElasticNet(selection = self.selection,
			max_iter = self.max_iter,
			l1_ratio = self.l1_ratio,
			copy_x = self.copy_x,
			normalize = self.normalize,
			random_state = self.random_state,
			warm_start = self.warm_start,
			fit_intercept = self.fit_intercept,
			alpha = self.alpha,
			positive = self.positive,
			precompute = self.precompute,
			tol = self.tol)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(check_input=check_input,
			X=X,
			sample_weight=sample_weight,
			y=y)

