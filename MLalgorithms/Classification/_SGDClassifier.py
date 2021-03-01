
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(intercept_init=intercept_init,
			coef_init=coef_init,
			y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.epsilon = epsilon
		self.tol = tol
		self.verbose = verbose
		self.class_weight = class_weight
		self.fit_intercept = fit_intercept
		self.validation_fraction = validation_fraction
		self.eta0 = eta0
		self.random_state = random_state
		self.l1_ratio = l1_ratio
		self.learning_rate = learning_rate
		self.shuffle = shuffle
		self.warm_start = warm_start
		self.penalty = penalty
		self.alpha = alpha
		self.n_iter_no_change = n_iter_no_change
		self.power_t = power_t
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.loss = loss
		self.early_stopping = early_stopping
		self.average = average
		self.model = SGDC(penalty = self.penalty,
			validation_fraction = self.validation_fraction,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			eta0 = self.eta0,
			warm_start = self.warm_start,
			average = self.average,
			shuffle = self.shuffle,
			epsilon = self.epsilon,
			loss = self.loss,
			n_iter_no_change = self.n_iter_no_change,
			alpha = self.alpha,
			random_state = self.random_state,
			learning_rate = self.learning_rate,
			n_jobs = self.n_jobs,
			early_stopping = self.early_stopping,
			class_weight = self.class_weight,
			l1_ratio = self.l1_ratio,
			power_t = self.power_t,
			verbose = self.verbose,
			tol = self.tol)

	def predict(self, X):
		return self.model.predict(X=X)

