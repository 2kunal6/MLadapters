
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.verbose = verbose
		self.degree = degree
		self.tol = tol
		self.probability = probability
		self.cache_size = cache_size
		self.C = C
		self.class_weight = class_weight
		self.decision_function_shape = decision_function_shape
		self.random_state = random_state
		self.break_ties = break_ties
		self.max_iter = max_iter
		self.kernel = kernel
		self.gamma = gamma
		self.shrinking = shrinking
		self.coef0 = coef0
		self.model = SVCClassification(degree = self.degree,
			max_iter = self.max_iter,
			coef0 = self.coef0,
			cache_size = self.cache_size,
			probability = self.probability,
			class_weight = self.class_weight,
			tol = self.tol,
			break_ties = self.break_ties,
			random_state = self.random_state,
			decision_function_shape = self.decision_function_shape,
			gamma = self.gamma,
			C = self.C,
			shrinking = self.shrinking,
			verbose = self.verbose,
			kernel = self.kernel)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

