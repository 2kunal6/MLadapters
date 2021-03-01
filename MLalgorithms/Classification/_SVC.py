
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.class_weight = class_weight
		self.coef0 = coef0
		self.degree = degree
		self.break_ties = break_ties
		self.tol = tol
		self.cache_size = cache_size
		self.shrinking = shrinking
		self.kernel = kernel
		self.max_iter = max_iter
		self.probability = probability
		self.gamma = gamma
		self.verbose = verbose
		self.decision_function_shape = decision_function_shape
		self.random_state = random_state
		self.C = C
		self.model = SVCClassification(coef0 = self.coef0,
			shrinking = self.shrinking,
			degree = self.degree,
			tol = self.tol,
			decision_function_shape = self.decision_function_shape,
			kernel = self.kernel,
			gamma = self.gamma,
			verbose = self.verbose,
			cache_size = self.cache_size,
			probability = self.probability,
			random_state = self.random_state,
			C = self.C,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			break_ties = self.break_ties)

