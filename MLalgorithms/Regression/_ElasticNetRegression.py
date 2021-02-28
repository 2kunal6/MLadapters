
from sklearn.linear_model import ElasticNet
from MLalgorithms._Regression import Regression


class ElasticNetRegression(Regression):
	
	def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_x=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
		self.l1_ratio = l1_ratio
		self.positive = positive
		self.max_iter = max_iter
		self.selection = selection
		self.precompute = precompute
		self.warm_start = warm_start
		self.alpha = alpha
		self.random_state = random_state
		self.tol = tol
		Regression.__init__(self, fit_intercept=fit_intercept, normalize=normalize, copy_x=copy_x)
		self.model = ElasticNet(random_state = self.random_state,
			max_iter = self.max_iter,
			precompute = self.precompute,
			selection = self.selection,
			positive = self.positive,
			warm_start = self.warm_start,
			tol = self.tol,
			l1_ratio = self.l1_ratio,
			alpha = self.alpha,
			copy_x = self.copy_x,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None, check_input=True):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X,
			check_input=check_input)

