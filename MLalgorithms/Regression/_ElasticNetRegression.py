
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_x=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.precompute = precompute
		self.random_state = random_state
		self.tol = tol
		self.positive = positive
		self.selection = selection
		self.l1_ratio = l1_ratio
		self.max_iter = max_iter
		self.warm_start = warm_start
		self.alpha = alpha
		Regression.__init__(self, fit_intercept=fit_intercept, copy_x=copy_x, normalize=normalize)
		self.model = ElasticNet(l1_ratio = self.l1_ratio,
			normalize = self.normalize,
			max_iter = self.max_iter,
			tol = self.tol,
			positive = self.positive,
			selection = self.selection,
			fit_intercept = self.fit_intercept,
			copy_x = self.copy_x,
			precompute = self.precompute,
			alpha = self.alpha,
			random_state = self.random_state,
			warm_start = self.warm_start)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(X=X,
			y=y,
			sample_weight=sample_weight,
			check_input=check_input)

	def predict(self, X):
		return self.model.predict(X=X)

