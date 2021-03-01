
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.class_weight = class_weight
		self.tol = tol
		self.alpha = alpha
		self.early_stopping = early_stopping
		self.max_iter = max_iter
		self.penalty = penalty
		self.fit_intercept = fit_intercept
		self.validation_fraction = validation_fraction
		self.random_state = random_state
		self.verbose = verbose
		self.average = average
		self.l1_ratio = l1_ratio
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.warm_start = warm_start
		self.n_jobs = n_jobs
		self.shuffle = shuffle
		self.eta0 = eta0
		self.n_iter_no_change = n_iter_no_change
		self.power_t = power_t
		self.loss = loss
		self.model = SGDC(power_t = self.power_t,
			random_state = self.random_state,
			average = self.average,
			n_iter_no_change = self.n_iter_no_change,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			penalty = self.penalty,
			early_stopping = self.early_stopping,
			n_jobs = self.n_jobs,
			shuffle = self.shuffle,
			learning_rate = self.learning_rate,
			alpha = self.alpha,
			tol = self.tol,
			loss = self.loss,
			verbose = self.verbose,
			warm_start = self.warm_start,
			epsilon = self.epsilon,
			l1_ratio = self.l1_ratio,
			eta0 = self.eta0,
			max_iter = self.max_iter,
			validation_fraction = self.validation_fraction)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(X=X,
			coef_init=coef_init,
			y=y,
			intercept_init=intercept_init,
			sample_weight=sample_weight)

