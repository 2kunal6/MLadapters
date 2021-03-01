
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier as SGDC
from MLalgorithms._Classification import Classification


class SGDClassifier(Classification):
	
	def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False):
		self.n_iter_no_change = n_iter_no_change
		self.fit_intercept = fit_intercept
		self.early_stopping = early_stopping
		self.l1_ratio = l1_ratio
		self.penalty = penalty
		self.epsilon = epsilon
		self.shuffle = shuffle
		self.random_state = random_state
		self.average = average
		self.power_t = power_t
		self.verbose = verbose
		self.validation_fraction = validation_fraction
		self.max_iter = max_iter
		self.loss = loss
		self.tol = tol
		self.class_weight = class_weight
		self.n_jobs = n_jobs
		self.alpha = alpha
		self.eta0 = eta0
		self.learning_rate = learning_rate
		self.warm_start = warm_start
		self.model = SGDC(random_state = self.random_state,
			class_weight = self.class_weight,
			fit_intercept = self.fit_intercept,
			power_t = self.power_t,
			learning_rate = self.learning_rate,
			verbose = self.verbose,
			validation_fraction = self.validation_fraction,
			loss = self.loss,
			n_iter_no_change = self.n_iter_no_change,
			eta0 = self.eta0,
			epsilon = self.epsilon,
			alpha = self.alpha,
			l1_ratio = self.l1_ratio,
			tol = self.tol,
			penalty = self.penalty,
			average = self.average,
			early_stopping = self.early_stopping,
			max_iter = self.max_iter,
			warm_start = self.warm_start,
			n_jobs = self.n_jobs,
			shuffle = self.shuffle)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None):
		return self.model.fit(X=X,
			intercept_init=intercept_init,
			coef_init=coef_init,
			y=y,
			sample_weight=sample_weight)

