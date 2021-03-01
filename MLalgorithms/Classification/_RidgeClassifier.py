
from sklearn.linear_model import RidgeClassifier as RC
from MLalgorithms._Classification import Classification


class RidgeClassifier(Classification):
	
	def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_x=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
		self.solver = solver
		self.max_iter = max_iter
		self.random_state = random_state
		self.alpha = alpha
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.copy_x = copy_x
		self.class_weight = class_weight
		self.model = RC(class_weight = self.class_weight,
			normalize = self.normalize,
			copy_x = self.copy_x,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			tol = self.tol,
			random_state = self.random_state,
			alpha = self.alpha,
			solver = self.solver)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

