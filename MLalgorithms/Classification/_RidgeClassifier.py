
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.random_state = random_state
		self.class_weight = class_weight
		self.tol = tol
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.solver = solver
		self.copy_x = copy_x
		self.normalize = normalize
		self.alpha = alpha
		self.model = RC(tol = self.tol,
			normalize = self.normalize,
			random_state = self.random_state,
			copy_x = self.copy_x,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			solver = self.solver,
			fit_intercept = self.fit_intercept,
			alpha = self.alpha)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

