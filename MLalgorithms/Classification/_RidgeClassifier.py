
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.random_state = random_state
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.solver = solver
		self.tol = tol
		self.alpha = alpha
		self.normalize = normalize
		self.copy_x = copy_x
		self.class_weight = class_weight
		self.model = RC(normalize = self.normalize,
			random_state = self.random_state,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			copy_x = self.copy_x,
			solver = self.solver,
			tol = self.tol,
			alpha = self.alpha)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

