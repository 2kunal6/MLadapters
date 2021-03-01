
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.class_weight = class_weight
		self.alpha = alpha
		self.solver = solver
		self.copy_x = copy_x
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.tol = tol
		self.random_state = random_state
		self.model = RC(normalize = self.normalize,
			alpha = self.alpha,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			copy_x = self.copy_x,
			random_state = self.random_state,
			solver = self.solver)

