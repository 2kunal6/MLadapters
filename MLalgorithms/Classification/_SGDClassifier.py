
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.warm_start = warm_start
		self.penalty = penalty
		self.epsilon = epsilon
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.learning_rate = learning_rate
		self.verbose = verbose
		self.eta0 = eta0
		self.early_stopping = early_stopping
		self.n_jobs = n_jobs
		self.class_weight = class_weight
		self.tol = tol
		self.validation_fraction = validation_fraction
		self.l1_ratio = l1_ratio
		self.power_t = power_t
		self.alpha = alpha
		self.n_iter_no_change = n_iter_no_change
		self.max_iter = max_iter
		self.loss = loss
		self.average = average
		self.shuffle = shuffle
		self.model = SGDC(warm_start = self.warm_start,
			learning_rate = self.learning_rate,
			verbose = self.verbose,
			early_stopping = self.early_stopping,
			l1_ratio = self.l1_ratio,
			max_iter = self.max_iter,
			validation_fraction = self.validation_fraction,
			random_state = self.random_state,
			fit_intercept = self.fit_intercept,
			average = self.average,
			tol = self.tol,
			epsilon = self.epsilon,
			shuffle = self.shuffle,
			n_jobs = self.n_jobs,
			eta0 = self.eta0,
			n_iter_no_change = self.n_iter_no_change,
			class_weight = self.class_weight,
			penalty = self.penalty,
			alpha = self.alpha,
			power_t = self.power_t,
			loss = self.loss)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(X=X,
			intercept_init=intercept_init,
			coef_init=coef_init,
			sample_weight=sample_weight,
			y=y)

