
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.class_weight = class_weight
		self.gamma = gamma
		self.verbose = verbose
		self.decision_function_shape = decision_function_shape
		self.kernel = kernel
		self.probability = probability
		self.shrinking = shrinking
		self.random_state = random_state
		self.cache_size = cache_size
		self.degree = degree
		self.tol = tol
		self.break_ties = break_ties
		self.coef0 = coef0
		self.max_iter = max_iter
		self.C = C
		self.model = SVCClassification(degree = self.degree,
			verbose = self.verbose,
			coef0 = self.coef0,
			shrinking = self.shrinking,
			kernel = self.kernel,
			break_ties = self.break_ties,
			gamma = self.gamma,
			probability = self.probability,
			decision_function_shape = self.decision_function_shape,
			class_weight = self.class_weight,
			max_iter = self.max_iter,
			tol = self.tol,
			random_state = self.random_state,
			C = self.C,
			cache_size = self.cache_size)

