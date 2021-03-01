
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(y=y,
			intercept_init=intercept_init,
			coef_init=coef_init,
			X=X,
			sample_weight=sample_weight)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.verbose = verbose
		self.alpha = alpha
		self.loss = loss
		self.eta0 = eta0
		self.l1_ratio = l1_ratio
		self.epsilon = epsilon
		self.n_iter_no_change = n_iter_no_change
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.tol = tol
		self.validation_fraction = validation_fraction
		self.shuffle = shuffle
		self.random_state = random_state
		self.penalty = penalty
		self.average = average
		self.class_weight = class_weight
		self.power_t = power_t
		self.early_stopping = early_stopping
		self.n_jobs = n_jobs
		self.learning_rate = learning_rate
		self.fit_intercept = fit_intercept
		self.model = SGDC(loss = self.loss,
			class_weight = self.class_weight,
			n_iter_no_change = self.n_iter_no_change,
			penalty = self.penalty,
			eta0 = self.eta0,
			random_state = self.random_state,
			alpha = self.alpha,
			tol = self.tol,
			learning_rate = self.learning_rate,
			shuffle = self.shuffle,
			l1_ratio = self.l1_ratio,
			average = self.average,
			validation_fraction = self.validation_fraction,
			verbose = self.verbose,
			power_t = self.power_t,
			epsilon = self.epsilon,
			max_iter = self.max_iter,
			early_stopping = self.early_stopping,
			n_jobs = self.n_jobs,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start)

