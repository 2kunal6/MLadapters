
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.coef0 = coef0
		self.cache_size = cache_size
		self.break_ties = break_ties
		self.degree = degree
		self.max_iter = max_iter
		self.C = C
		self.kernel = kernel
		self.shrinking = shrinking
		self.verbose = verbose
		self.random_state = random_state
		self.tol = tol
		self.probability = probability
		self.gamma = gamma
		self.decision_function_shape = decision_function_shape
		self.class_weight = class_weight
		self.model = SVCClassification(break_ties = self.break_ties,
			decision_function_shape = self.decision_function_shape,
			class_weight = self.class_weight,
			shrinking = self.shrinking,
			cache_size = self.cache_size,
			verbose = self.verbose,
			max_iter = self.max_iter,
			gamma = self.gamma,
			tol = self.tol,
			coef0 = self.coef0,
			random_state = self.random_state,
			probability = self.probability,
			C = self.C,
			degree = self.degree,
			kernel = self.kernel)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

