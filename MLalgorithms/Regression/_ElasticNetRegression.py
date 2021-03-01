
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			check_input=check_input,
			X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.positive = positive
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.tol = tol
		self.random_state = random_state
		self.selection = selection
		self.precompute = precompute
		self.l1_ratio = l1_ratio
		Regression.__init__(self, fit_intercept=fit_intercept, copy_X=copy_X, normalize=normalize)
		self.model = ElasticNet(l1_ratio = self.l1_ratio,
			alpha = self.alpha,
			warm_start = self.warm_start,
			copy_X = self.copy_X,
			normalize = self.normalize,
			positive = self.positive,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			precompute = self.precompute,
			selection = self.selection,
			max_iter = self.max_iter,
			tol = self.tol)

	def predict(self, X):
		return self.model.predict(X=X)

