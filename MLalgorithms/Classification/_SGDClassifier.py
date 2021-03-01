
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.learning_rate = learning_rate
		self.power_t = power_t
		self.warm_start = warm_start
		self.early_stopping = early_stopping
		self.epsilon = epsilon
		self.fit_intercept = fit_intercept
		self.shuffle = shuffle
		self.penalty = penalty
		self.loss = loss
		self.eta0 = eta0
		self.average = average
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.validation_fraction = validation_fraction
		self.verbose = verbose
		self.tol = tol
		self.l1_ratio = l1_ratio
		self.n_iter_no_change = n_iter_no_change
		self.alpha = alpha
		self.model = SGDC(random_state = self.random_state,
			power_t = self.power_t,
			alpha = self.alpha,
			class_weight = self.class_weight,
			verbose = self.verbose,
			epsilon = self.epsilon,
			average = self.average,
			tol = self.tol,
			early_stopping = self.early_stopping,
			n_jobs = self.n_jobs,
			warm_start = self.warm_start,
			loss = self.loss,
			shuffle = self.shuffle,
			n_iter_no_change = self.n_iter_no_change,
			fit_intercept = self.fit_intercept,
			validation_fraction = self.validation_fraction,
			eta0 = self.eta0,
			learning_rate = self.learning_rate,
			l1_ratio = self.l1_ratio,
			penalty = self.penalty,
			max_iter = self.max_iter)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(X=X,
			y=y,
			intercept_init=intercept_init,
			sample_weight=sample_weight,
			coef_init=coef_init)

