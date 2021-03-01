
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.C = C
		self.shrinking = shrinking
		self.random_state = random_state
		self.verbose = verbose
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.cache_size = cache_size
		self.tol = tol
		self.break_ties = break_ties
		self.coef0 = coef0
		self.degree = degree
		self.kernel = kernel
		self.gamma = gamma
		self.probability = probability
		self.decision_function_shape = decision_function_shape
		self.model = SVCClassification(tol = self.tol,
			coef0 = self.coef0,
			C = self.C,
			degree = self.degree,
			gamma = self.gamma,
			shrinking = self.shrinking,
			probability = self.probability,
			random_state = self.random_state,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			verbose = self.verbose,
			kernel = self.kernel,
			cache_size = self.cache_size,
			decision_function_shape = self.decision_function_shape,
			break_ties = self.break_ties)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

