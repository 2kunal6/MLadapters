
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.warm_start = warm_start
		self.power_t = power_t
		self.learning_rate = learning_rate
		self.n_jobs = n_jobs
		self.penalty = penalty
		self.validation_fraction = validation_fraction
		self.random_state = random_state
		self.n_iter_no_change = n_iter_no_change
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.l1_ratio = l1_ratio
		self.verbose = verbose
		self.alpha = alpha
		self.shuffle = shuffle
		self.loss = loss
		self.eta0 = eta0
		self.average = average
		self.early_stopping = early_stopping
		self.tol = tol
		self.fit_intercept = fit_intercept
		self.epsilon = epsilon
		self.model = SGDC(penalty = self.penalty,
			l1_ratio = self.l1_ratio,
			shuffle = self.shuffle,
			warm_start = self.warm_start,
			tol = self.tol,
			epsilon = self.epsilon,
			n_iter_no_change = self.n_iter_no_change,
			loss = self.loss,
			eta0 = self.eta0,
			average = self.average,
			learning_rate = self.learning_rate,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			validation_fraction = self.validation_fraction,
			alpha = self.alpha,
			early_stopping = self.early_stopping,
			verbose = self.verbose,
			max_iter = self.max_iter,
			power_t = self.power_t,
			random_state = self.random_state,
			n_jobs = self.n_jobs)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(intercept_init=intercept_init,
			coef_init=coef_init,
			sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

