
from sklearn.linear_model import LogisticRegression as LR
from MLalgorithms._Regression import Regression


class LogisticRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

	def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
		self.tol = tol
		self.verbose = verbose
		self.class_weight = class_weight
		self.multi_class = multi_class
		self.C = C
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.dual = dual
		self.solver = solver
		self.n_jobs = n_jobs
		self.intercept_scaling = intercept_scaling
		self.penalty = penalty
		self.random_state = random_state
		self.fit_intercept = fit_intercept
		self.l1_ratio = l1_ratio
		self.model = LR(dual = self.dual,
			l1_ratio = self.l1_ratio,
			penalty = self.penalty,
			multi_class = self.multi_class,
			max_iter = self.max_iter,
			verbose = self.verbose,
			class_weight = self.class_weight,
			intercept_scaling = self.intercept_scaling,
			random_state = self.random_state,
			tol = self.tol,
			solver = self.solver,
			C = self.C,
			n_jobs = self.n_jobs,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start)

