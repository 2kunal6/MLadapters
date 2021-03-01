
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.l1_ratio = l1_ratio
		self.loss = loss
		self.penalty = penalty
		self.verbose = verbose
		self.learning_rate = learning_rate
		self.average = average
		self.epsilon = epsilon
		self.tol = tol
		self.max_iter = max_iter
		self.shuffle = shuffle
		self.class_weight = class_weight
		self.random_state = random_state
		self.alpha = alpha
		self.warm_start = warm_start
		self.power_t = power_t
		self.early_stopping = early_stopping
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.validation_fraction = validation_fraction
		self.eta0 = eta0
		self.n_iter_no_change = n_iter_no_change
		self.model = SGDC(penalty = self.penalty,
			average = self.average,
			alpha = self.alpha,
			l1_ratio = self.l1_ratio,
			n_iter_no_change = self.n_iter_no_change,
			loss = self.loss,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			validation_fraction = self.validation_fraction,
			eta0 = self.eta0,
			epsilon = self.epsilon,
			early_stopping = self.early_stopping,
			tol = self.tol,
			random_state = self.random_state,
			shuffle = self.shuffle,
			n_jobs = self.n_jobs,
			verbose = self.verbose,
			max_iter = self.max_iter,
			warm_start = self.warm_start,
			power_t = self.power_t,
			learning_rate = self.learning_rate)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(intercept_init=intercept_init,
			sample_weight=sample_weight,
			X=X,
			coef_init=coef_init,
			y=y)

