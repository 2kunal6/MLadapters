
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV as LRCV
from MLalgorithms._Regression import Regression


class LogisticRegressionCV(Regression):
	
	def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1, multi_class='auto', random_state=None, l1_ratios=None):
		self.multi_class = multi_class
		self.intercept_scaling = intercept_scaling
		self.cv = cv
		self.Cs = Cs
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.penalty = penalty
		self.random_state = random_state
		self.scoring = scoring
		self.verbose = verbose
		self.dual = dual
		self.n_jobs = n_jobs
		self.refit = refit
		self.tol = tol
		self.l1_ratios = l1_ratios
		self.solver = solver
		self.fit_intercept = fit_intercept
		self.model = LRCV(intercept_scaling = self.intercept_scaling,
			max_iter = self.max_iter,
			penalty = self.penalty,
			refit = self.refit,
			l1_ratios = self.l1_ratios,
			class_weight = self.class_weight,
			dual = self.dual,
			fit_intercept = self.fit_intercept,
			solver = self.solver,
			cv = self.cv,
			tol = self.tol,
			random_state = self.random_state,
			n_jobs = self.n_jobs,
			verbose = self.verbose,
			scoring = self.scoring,
			multi_class = self.multi_class,
			Cs = self.Cs)

	def predict(self, X):
		return self.model.predict(X=X)

	def fit(self, X, y, sample_weight=None):
		return self.model.fit(sample_weight=sample_weight,
			y=y,
			X=X)

