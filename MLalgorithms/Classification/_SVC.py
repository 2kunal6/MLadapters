
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.tol = tol
		self.class_weight = class_weight
		self.shrinking = shrinking
		self.C = C
		self.verbose = verbose
		self.degree = degree
		self.coef0 = coef0
		self.decision_function_shape = decision_function_shape
		self.kernel = kernel
		self.cache_size = cache_size
		self.probability = probability
		self.break_ties = break_ties
		self.random_state = random_state
		self.gamma = gamma
		self.max_iter = max_iter
		self.model = SVCClassification(shrinking = self.shrinking,
			break_ties = self.break_ties,
			decision_function_shape = self.decision_function_shape,
			kernel = self.kernel,
			degree = self.degree,
			max_iter = self.max_iter,
			verbose = self.verbose,
			class_weight = self.class_weight,
			random_state = self.random_state,
			tol = self.tol,
			gamma = self.gamma,
			C = self.C,
			cache_size = self.cache_size,
			coef0 = self.coef0,
			probability = self.probability)

