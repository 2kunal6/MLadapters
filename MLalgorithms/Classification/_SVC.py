
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.verbose = verbose
		self.random_state = random_state
		self.class_weight = class_weight
		self.coef0 = coef0
		self.kernel = kernel
		self.shrinking = shrinking
		self.degree = degree
		self.break_ties = break_ties
		self.probability = probability
		self.cache_size = cache_size
		self.tol = tol
		self.decision_function_shape = decision_function_shape
		self.C = C
		self.gamma = gamma
		self.max_iter = max_iter
		self.model = SVCClassification(shrinking = self.shrinking,
			max_iter = self.max_iter,
			break_ties = self.break_ties,
			verbose = self.verbose,
			coef0 = self.coef0,
			tol = self.tol,
			kernel = self.kernel,
			decision_function_shape = self.decision_function_shape,
			C = self.C,
			gamma = self.gamma,
			probability = self.probability,
			cache_size = self.cache_size,
			class_weight = self.class_weight,
			random_state = self.random_state,
			degree = self.degree)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			y=y,
			sample_weight=sample_weight)

	def predict(self, X):
		return self.model.predict(X=X)

