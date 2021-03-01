
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
		self.normalize = normalize
		self.tol = tol
		self.solver = solver
		self.class_weight = class_weight
		self.max_iter = max_iter
		self.alpha = alpha
		self.copy_x = copy_x
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.model = RC(max_iter = self.max_iter,
			class_weight = self.class_weight,
			copy_x = self.copy_x,
			alpha = self.alpha,
			random_state = self.random_state,
			tol = self.tol,
			solver = self.solver,
			fit_intercept = self.fit_intercept,
			normalize = self.normalize)

