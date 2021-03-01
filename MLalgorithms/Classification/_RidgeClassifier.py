
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.tol = tol
		self.normalize = normalize
		self.class_weight = class_weight
		self.random_state = random_state
		self.alpha = alpha
		self.copy_x = copy_x
		self.max_iter = max_iter
		self.solver = solver
		self.fit_intercept = fit_intercept
		self.model = RC(max_iter = self.max_iter,
			normalize = self.normalize,
			alpha = self.alpha,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			solver = self.solver,
			tol = self.tol,
			random_state = self.random_state,
			copy_x = self.copy_x)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

