
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(coef_init=coef_init,
			X=X,
			sample_weight=sample_weight,
			intercept_init=intercept_init,
			y=y)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.n_jobs = n_jobs
		self.epsilon = epsilon
		self.fit_intercept = fit_intercept
		self.l1_ratio = l1_ratio
		self.shuffle = shuffle
		self.warm_start = warm_start
		self.validation_fraction = validation_fraction
		self.average = average
		self.early_stopping = early_stopping
		self.loss = loss
		self.tol = tol
		self.class_weight = class_weight
		self.learning_rate = learning_rate
		self.verbose = verbose
		self.alpha = alpha
		self.eta0 = eta0
		self.n_iter_no_change = n_iter_no_change
		self.power_t = power_t
		self.max_iter = max_iter
		self.penalty = penalty
		self.random_state = random_state
		self.model = SGDC(penalty = self.penalty,
			random_state = self.random_state,
			validation_fraction = self.validation_fraction,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			epsilon = self.epsilon,
			tol = self.tol,
			shuffle = self.shuffle,
			n_jobs = self.n_jobs,
			l1_ratio = self.l1_ratio,
			warm_start = self.warm_start,
			verbose = self.verbose,
			eta0 = self.eta0,
			early_stopping = self.early_stopping,
			n_iter_no_change = self.n_iter_no_change,
			average = self.average,
			loss = self.loss,
			alpha = self.alpha,
			learning_rate = self.learning_rate,
			max_iter = self.max_iter,
			power_t = self.power_t)

