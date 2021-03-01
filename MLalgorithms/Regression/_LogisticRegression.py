
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as LR
from MLalgorithms._Regression import Regression


class LogisticRegression(Regression):
	
	def predict(self, X):
		return self.model.predict(X=X)

	def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
		self.verbose = verbose
		self.multi_class = multi_class
		self.intercept_scaling = intercept_scaling
		self.warm_start = warm_start
		self.l1_ratio = l1_ratio
		self.fit_intercept = fit_intercept
		self.n_jobs = n_jobs
		self.tol = tol
		self.C = C
		self.solver = solver
		self.random_state = random_state
		self.penalty = penalty
		self.dual = dual
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.model = LR(random_state = self.random_state,
			tol = self.tol,
			warm_start = self.warm_start,
			class_weight = self.class_weight,
			l1_ratio = self.l1_ratio,
			n_jobs = self.n_jobs,
			dual = self.dual,
			penalty = self.penalty,
			fit_intercept = self.fit_intercept,
			max_iter = self.max_iter,
			intercept_scaling = self.intercept_scaling,
			verbose = self.verbose,
			solver = self.solver,
			multi_class = self.multi_class,
			C = self.C)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(y=y,
			sample_weight=sample_weight,
			X=X)

