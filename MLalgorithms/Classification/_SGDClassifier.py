
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.average = average
		self.fit_intercept = fit_intercept
		self.l1_ratio = l1_ratio
		self.learning_rate = learning_rate
		self.random_state = random_state
		self.loss = loss
		self.epsilon = epsilon
		self.alpha = alpha
		self.eta0 = eta0
		self.max_iter = max_iter
		self.tol = tol
		self.warm_start = warm_start
		self.n_iter_no_change = n_iter_no_change
		self.power_t = power_t
		self.n_jobs = n_jobs
		self.early_stopping = early_stopping
		self.class_weight = class_weight
		self.verbose = verbose
		self.validation_fraction = validation_fraction
		self.shuffle = shuffle
		self.penalty = penalty
		self.model = SGDC(early_stopping = self.early_stopping,
			loss = self.loss,
			tol = self.tol,
			random_state = self.random_state,
			validation_fraction = self.validation_fraction,
			eta0 = self.eta0,
			fit_intercept = self.fit_intercept,
			shuffle = self.shuffle,
			n_iter_no_change = self.n_iter_no_change,
			warm_start = self.warm_start,
			l1_ratio = self.l1_ratio,
			average = self.average,
			learning_rate = self.learning_rate,
			n_jobs = self.n_jobs,
			max_iter = self.max_iter,
			penalty = self.penalty,
			verbose = self.verbose,
			epsilon = self.epsilon,
			class_weight = self.class_weight,
			alpha = self.alpha,
			power_t = self.power_t)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(X=X,
			y=y,
			coef_init=coef_init,
			intercept_init=intercept_init,
			sample_weight=sample_weight)

	def predict(self, X):
		return self.model.predict(X=X)

