
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC as SVCClassification
from MLalgorithms._Classification import Classification


class SVC(Classification):
	
	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
		self.shrinking = shrinking
		self.verbose = verbose
		self.decision_function_shape = decision_function_shape
		self.probability = probability
		self.C = C
		self.tol = tol
		self.class_weight = class_weight
		self.cache_size = cache_size
		self.degree = degree
		self.coef0 = coef0
		self.random_state = random_state
		self.kernel = kernel
		self.max_iter = max_iter
		self.gamma = gamma
		self.break_ties = break_ties
		self.model = SVCClassification(class_weight = self.class_weight,
			probability = self.probability,
			shrinking = self.shrinking,
			kernel = self.kernel,
			C = self.C,
			break_ties = self.break_ties,
			decision_function_shape = self.decision_function_shape,
			random_state = self.random_state,
			gamma = self.gamma,
			degree = self.degree,
			verbose = self.verbose,
			max_iter = self.max_iter,
			tol = self.tol,
			coef0 = self.coef0,
			cache_size = self.cache_size)

	def predict(self, X):
		return self.model.predict(X=X)

