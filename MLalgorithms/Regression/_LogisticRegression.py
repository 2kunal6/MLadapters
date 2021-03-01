
from sklearn.linear_model import LogisticRegression as LR
from MLalgorithms._Regression import Regression


class LogisticRegression(Regression):
	
	def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
		self.dual = dual
		self.warm_start = warm_start
		self.multi_class = multi_class
		self.C = C
		self.solver = solver
		self.l1_ratio = l1_ratio
		self.intercept_scaling = intercept_scaling
		self.n_jobs = n_jobs
		self.penalty = penalty
		self.random_state = random_state
		self.verbose = verbose
		self.fit_intercept = fit_intercept
		self.tol = tol
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.model = LR(dual = self.dual,
			n_jobs = self.n_jobs,
			penalty = self.penalty,
			class_weight = self.class_weight,
			verbose = self.verbose,
			l1_ratio = self.l1_ratio,
			max_iter = self.max_iter,
			fit_intercept = self.fit_intercept,
			warm_start = self.warm_start,
			tol = self.tol,
			intercept_scaling = self.intercept_scaling,
			random_state = self.random_state,
			C = self.C,
			solver = self.solver,
			multi_class = self.multi_class)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			X=X,
			y=y)

	def predict(self, X):
		return self.model.predict(X=X)

