
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.degree = degree
		self.coef0 = coef0
		self.C = C
		self.cache_size = cache_size
		self.verbose = verbose
		self.gamma = gamma
		self.max_iter = max_iter
		self.kernel = kernel
		self.random_state = random_state
		self.decision_function_shape = decision_function_shape
		self.tol = tol
		self.class_weight = class_weight
		self.shrinking = shrinking
		self.probability = probability
		self.break_ties = break_ties
		self.model = SVCClassification(degree = self.degree,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			shrinking = self.shrinking,
			cache_size = self.cache_size,
			random_state = self.random_state,
			C = self.C,
			coef0 = self.coef0,
			verbose = self.verbose,
			kernel = self.kernel,
			decision_function_shape = self.decision_function_shape,
			break_ties = self.break_ties,
			probability = self.probability,
			tol = self.tol,
			gamma = self.gamma)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(X=X,
			sample_weight=sample_weight,
			y=y)

