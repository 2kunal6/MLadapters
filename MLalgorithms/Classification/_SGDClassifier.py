
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(coef_init=coef_init,
			X=X,
			sample_weight=sample_weight,
			y=y,
			intercept_init=intercept_init)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.alpha = alpha
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.early_stopping = early_stopping
		self.loss = loss
		self.verbose = verbose
		self.shuffle = shuffle
		self.tol = tol
		self.eta0 = eta0
		self.n_iter_no_change = n_iter_no_change
		self.average = average
		self.warm_start = warm_start
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.validation_fraction = validation_fraction
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.n_jobs = n_jobs
		self.l1_ratio = l1_ratio
		self.power_t = power_t
		self.penalty = penalty
		self.model = SGDC(tol = self.tol,
			learning_rate = self.learning_rate,
			shuffle = self.shuffle,
			n_iter_no_change = self.n_iter_no_change,
			validation_fraction = self.validation_fraction,
			random_state = self.random_state,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			eta0 = self.eta0,
			warm_start = self.warm_start,
			verbose = self.verbose,
			penalty = self.penalty,
			alpha = self.alpha,
			n_jobs = self.n_jobs,
			epsilon = self.epsilon,
			l1_ratio = self.l1_ratio,
			average = self.average,
			early_stopping = self.early_stopping,
			power_t = self.power_t,
			loss = self.loss,
			max_iter = self.max_iter)

