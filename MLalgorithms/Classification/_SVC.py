
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
		self.cache_size = cache_size
		self.decision_function_shape = decision_function_shape
		self.verbose = verbose
		self.tol = tol
		self.break_ties = break_ties
		self.coef0 = coef0
		self.shrinking = shrinking
		self.gamma = gamma
		self.class_weight = class_weight
		self.kernel = kernel
		self.max_iter = max_iter
		self.C = C
		self.probability = probability
		self.random_state = random_state
		self.degree = degree
		self.model = SVCClassification(C = self.C,
			degree = self.degree,
			shrinking = self.shrinking,
			max_iter = self.max_iter,
			coef0 = self.coef0,
			probability = self.probability,
			decision_function_shape = self.decision_function_shape,
			class_weight = self.class_weight,
			break_ties = self.break_ties,
			tol = self.tol,
			verbose = self.verbose,
			cache_size = self.cache_size,
			random_state = self.random_state,
			gamma = self.gamma,
			kernel = self.kernel)

