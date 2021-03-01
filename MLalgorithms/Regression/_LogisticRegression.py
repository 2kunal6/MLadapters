
from sklearn.linear_model import LogisticRegression as LR
from MLalgorithms._Regression import Regression


class LogisticRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

	def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
		self.class_weight = class_weight
		self.dual = dual
		self.solver = solver
		self.tol = tol
		self.verbose = verbose
		self.intercept_scaling = intercept_scaling
		self.n_jobs = n_jobs
		self.l1_ratio = l1_ratio
		self.warm_start = warm_start
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.penalty = penalty
		self.multi_class = multi_class
		self.random_state = random_state
		self.C = C
		self.model = LR(l1_ratio = self.l1_ratio,
			verbose = self.verbose,
			warm_start = self.warm_start,
			penalty = self.penalty,
			dual = self.dual,
			max_iter = self.max_iter,
			class_weight = self.class_weight,
			intercept_scaling = self.intercept_scaling,
			tol = self.tol,
			fit_intercept = self.fit_intercept,
			multi_class = self.multi_class,
			random_state = self.random_state,
			C = self.C,
			solver = self.solver,
			n_jobs = self.n_jobs)

