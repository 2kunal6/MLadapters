
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X,
			check_input=check_input)

	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.max_iter = max_iter
		self.precompute = precompute
		self.random_state = random_state
		self.l1_ratio = l1_ratio
		self.selection = selection
		self.warm_start = warm_start
		self.positive = positive
		self.tol = tol
		self.alpha = alpha
		Regression.__init__(self, normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
		self.model = ElasticNet(normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			copy_X = self.copy_X,
			positive = self.positive,
			alpha = self.alpha,
			warm_start = self.warm_start,
			max_iter = self.max_iter,
			precompute = self.precompute,
			selection = self.selection,
			l1_ratio = self.l1_ratio,
			tol = self.tol,
			random_state = self.random_state)

