
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(coef_init=coef_init,
			intercept_init=intercept_init,
			X=X,
			y=y,
			sample_weight=sample_weight)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.validation_fraction = validation_fraction
		self.early_stopping = early_stopping
		self.average = average
		self.eta0 = eta0
		self.warm_start = warm_start
		self.shuffle = shuffle
		self.verbose = verbose
		self.epsilon = epsilon
		self.power_t = power_t
		self.random_state = random_state
		self.alpha = alpha
		self.l1_ratio = l1_ratio
		self.loss = loss
		self.penalty = penalty
		self.max_iter = max_iter
		self.n_jobs = n_jobs
		self.tol = tol
		self.n_iter_no_change = n_iter_no_change
		self.class_weight = class_weight
		self.learning_rate = learning_rate
		self.fit_intercept = fit_intercept
		self.model = SGDC(n_iter_no_change = self.n_iter_no_change,
			warm_start = self.warm_start,
			epsilon = self.epsilon,
			alpha = self.alpha,
			early_stopping = self.early_stopping,
			n_jobs = self.n_jobs,
			verbose = self.verbose,
			random_state = self.random_state,
			loss = self.loss,
			shuffle = self.shuffle,
			learning_rate = self.learning_rate,
			penalty = self.penalty,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			power_t = self.power_t,
			average = self.average,
			l1_ratio = self.l1_ratio,
			eta0 = self.eta0,
			max_iter = self.max_iter,
			validation_fraction = self.validation_fraction,
			class_weight = self.class_weight)

