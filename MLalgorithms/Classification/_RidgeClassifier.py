
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.alpha = alpha
		self.solver = solver
		self.tol = tol
		self.class_weight = class_weight
		self.copy_x = copy_x
		self.fit_intercept = fit_intercept
		self.max_iter = max_iter
		self.random_state = random_state
		self.normalize = normalize
		self.model = RC(class_weight = self.class_weight,
			alpha = self.alpha,
			normalize = self.normalize,
			fit_intercept = self.fit_intercept,
			random_state = self.random_state,
			copy_x = self.copy_x,
			max_iter = self.max_iter,
			tol = self.tol,
			solver = self.solver)

	def predict(self, X):
		return self.model.predict(X=X)

