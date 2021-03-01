
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.max_iter = max_iter
		self.tol = tol
		self.normalize = normalize
		self.class_weight = class_weight
		self.random_state = random_state
		self.copy_x = copy_x
		self.fit_intercept = fit_intercept
		self.alpha = alpha
		self.solver = solver
		self.model = RC(fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			alpha = self.alpha,
			copy_x = self.copy_x,
			normalize = self.normalize,
			solver = self.solver,
			tol = self.tol,
			class_weight = self.class_weight,
			random_state = self.random_state)

