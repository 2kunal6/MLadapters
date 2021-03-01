
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.tol = tol
		self.max_iter = max_iter
		self.cache_size = cache_size
		self.gamma = gamma
		self.shrinking = shrinking
		self.kernel = kernel
		self.probability = probability
		self.degree = degree
		self.C = C
		self.coef0 = coef0
		self.decision_function_shape = decision_function_shape
		self.random_state = random_state
		self.verbose = verbose
		self.class_weight = class_weight
		self.break_ties = break_ties
		self.model = SVCClassification(decision_function_shape = self.decision_function_shape,
			coef0 = self.coef0,
			random_state = self.random_state,
			tol = self.tol,
			class_weight = self.class_weight,
			cache_size = self.cache_size,
			degree = self.degree,
			probability = self.probability,
			gamma = self.gamma,
			max_iter = self.max_iter,
			verbose = self.verbose,
			break_ties = self.break_ties,
			C = self.C,
			shrinking = self.shrinking,
			kernel = self.kernel)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

